import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import third_party

sys.path.append('../../')
import clip
import timm
from train.prompt_templates import spoof_templates, real_templates
from collections import OrderedDict
from clip.model import CLIP

from utils.rkd_loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer
from utils.distkd_loss import DIST


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# feature generators
class feature_generator_adapt(nn.Module):

    def __init__(self, gamma, beta):
      super(feature_generator_adapt, self).__init__()
      self.vit = third_party.create_model(
          'vit_base_patch16_224', pretrained=True, gamma=gamma, beta=beta)

    def forward(self, input):
      feat, total_loss = self.vit.forward_features(input)
      return feat, total_loss


class feature_generator(nn.Module):

    def __init__(self):
      super(feature_generator, self).__init__()
      self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
      # self.vit.head = nn.Identity() # remove the classification head for timm version 0.6.12

    def forward(self, input):
      # feat = self.vit.forward_features(input).detach()
      feat = self.vit.forward_features(input) # for timm version 0.4.9
      # feat = self.vit.forward_features(input) # for timm version 0.4.9
      # feat = self.vit.forward(input) # for timm version 0.6.12
      print(f'feat : {feat}')
      return feat


class feature_generator_beit(nn.Module):

    def __init__(self):
      super(feature_generator_beit, self).__init__()
      self.vit = timm.create_model('beitv2_base_patch16_224', pretrained=True) # This will load the v2 of BEiT and the weights will be the pretrained on in1k and intermediate finetuned on 21kto1k
      self.vit.head = nn.Identity()

    def forward(self, input):
      feat = self.vit(input)
      return feat


class feature_generator_clip(nn.Module):

    def __init__(self):
      super(feature_generator_clip, self).__init__()
      # self.vit, _ = clip.load("ViT-B/16", device='cuda')
      self.vit = self.vit.visual

    def forward(self, input):
      feat = self.vit.forward_full(input)
      # feat, _ = self.vit.forward_full(input)
      return feat


class feature_generator_fix_vitaf(nn.Module):

    def __init__(self):
      super(feature_generator_fix_vitaf, self).__init__()
      self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
      self.vit.head = nn.Identity() # remove the classification head for timm version 0.6.12

    def forward(self, input):
      # feat = self.vit.forward_features(input).detach() # for timm version 0.4.9
      feat = self.vit.forward(input).detach() # for timm version 0.6.12
      # feat = self.vit.forward_features(input)
      return feat


# feature embedders
class feature_embedder(nn.Module):

    def __init__(self):
      super(feature_embedder, self).__init__()
      self.bottleneck_layer_fc = nn.Linear(768, 512)
      self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
      self.bottleneck_layer_fc.bias.data.fill_(0.1)
      self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(),
                                            nn.Dropout(0.5))

    def forward(self, input, norm_flag=True):
      feature = self.bottleneck_layer(input)
      if (norm_flag):
        feature_norm = torch.norm(
            feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)**0.5 * (2)**0.5
        feature = torch.div(feature, feature_norm)
      return feature

# classifier
class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if (norm_flag):
            self.classifier_layer.weight.data = l2_norm(
                self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

# loss function
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


# FLIP-MCL
class flip_mcl(nn.Module):

    def __init__(self, args, device, in_dim, ssl_mlp_dim, ssl_emb_dim):
        super(flip_mcl, self).__init__()
        # load the model

        self.args = args
        self.device = device

        self.t_model, _ = clip.load("ViT-B/16", 'cuda:0')        
        self.model = CLIP(512, 224, 12, 192, 16, 77, 49408, 384, 6, 12, args.swin)

        if args.t_embed_dim != args.s_embed_dim:
            self.visual_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)
            self.text_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)

        # define the SSL parameters
        self.image_mlp = self._build_mlp(in_dim=in_dim, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)
        self.n_views = 2
        self.temperature = 0.1

        # dot product similarity
        self.cosine_similarity = nn.CosineSimilarity()
        # mse loss
        self.mse_loss = nn.MSELoss()

        # loss function
        self.kl_loss = DistillKL(T=1)
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = {}


    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.BatchNorm1d(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.BatchNorm1d(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))
    
    def info_nce_loss(self, feature_view_1, feature_view_2):
            assert feature_view_1.shape == feature_view_2.shape
            features = torch.cat([feature_view_1, feature_view_2], dim=0)

            labels = torch.cat([torch.arange(feature_view_1.shape[0]) for i in range(self.n_views)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.cuda()

            features = F.normalize(features, dim=1)

            similarity_matrix = torch.matmul(features, features.T)

            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

            # select and combine multiple positives
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

            # select only the negatives the negatives
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            logits = logits / self.temperature
            return logits, labels
    
    def load_tmodel(self):
        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False

        ckpt = torch.load(self.args.t_model_checkpoint)
        state_dict = ckpt['state_dict']

        _state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '')
            _state_dict[new_key] = v

        model_keys = set(self.t_model.state_dict().keys())
        f_state_dict = {k: v for k, v in _state_dict.items() if k in model_keys}

        self.t_model.load_state_dict(f_state_dict)
        epoch = ckpt['epoch']
        iter_num_start = epoch*100

        self.t_model.eval()

        return self.t_model
    
    def get_grad(self, p, k, tau, targets):
        logits = p @ k.T / tau
        targets = F.one_hot(targets, num_classes=logits.size(1)).float()
        prob = F.softmax(logits, 1)
        grad_p = (prob - targets) @ k / tau / targets.size(0)
        embed_size = p.size(1)
        prob_targets_repeat = (prob - targets).t().repeat(1, embed_size).view(-1,embed_size, p.size(0))
        grad_k = (prob_targets_repeat * (p.t() / tau).unsqueeze(0)).sum(-1) / targets.size(0)

        return grad_p, grad_k

    def forward(self, input, input_view_1, input_view_2, source_labels, norm_flag=True):
        self.model.train()

        # -----------------------------------------t_model load--------------------------------------
        self.load_tmodel()
        
        #print(f'Loaded checkpoint from epoch {epoch} at iteration : {iter_num_start}' )
        #print('Teacher model loaded successfully')
        #----------------------------------------------------------------------------------------------

        # tokenize the spoof and real templates
        spoof_texts = clip.tokenize(spoof_templates).cuda(non_blocking=True) #tokenize
        real_texts = clip.tokenize(real_templates).cuda(non_blocking=True) #tokenize
        # encode the spoof and real templates with the text encoder
        all_spoof_class_embeddings = self.model.encode_text(spoof_texts)
        all_real_class_embeddings = self.model.encode_text(real_texts)

        with torch.no_grad():
            t_all_spoof_class_embeddings = self.t_model.encode_text(spoof_texts)
            t_all_real_class_embeddings = self.t_model.encode_text(real_texts)
        
        # ------------------- Image-Text similarity branch -------------------
        # Ensemble of text features
        # embed with text encoder
        spoof_class_embeddings = all_spoof_class_embeddings.mean(dim=0)
        real_class_embeddings = all_real_class_embeddings.mean(dim=0)

        t_spoof_class_embeddings = t_all_spoof_class_embeddings.mean(dim=0)
        t_real_class_embeddings = t_all_real_class_embeddings.mean(dim=0)

        # stack the embeddings for image-text similarity
        ensemble_weights = [spoof_class_embeddings, real_class_embeddings] 
        t_ensemble_weights = [t_spoof_class_embeddings, t_real_class_embeddings]
        text_features = torch.stack(ensemble_weights, dim=0).cuda()
        t_text_features = torch.stack(t_ensemble_weights, dim=0).cuda()

        # get the image features
        image_features, attention_weight = self.model.encode_image(input)
        if self.args.swin == True:
            batch_size, h, w, c = image_features.shape
            image_features = image_features.view(batch_size, h * w, c)

        with torch.no_grad():
            t_image_features, t_attention_weight = self.t_model.encode_image(input)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        t_logit_scale = self.t_model.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        t_logits_per_image = t_logit_scale * t_image_features @ t_text_features.t()
        t_logits_per_text = t_logits_per_image.T

        logits_per_s_image_to_t_text = self.cross_logit_scale * image_features @ t_text_features.T
        logits_per_s_text_to_t_image = self.cross_logit_scale * text_features @ t_image_features.T

        similarity = logits_per_image
        # ------------------- Image-Text similarity branch -------------------


        # ------------------------------ Image SSL branch -------------------------------- # 
        # Get the image embeddings for the ssl views
        aug1, _ = self.model.encode_image(input_view_1) # Bx512
        if self.args.swin == True:
            batch_size, h, w, c = aug1.shape
            aug1 = aug1.view(batch_size, h * w, c)
        aug2, _ = self.model.encode_image(input_view_2) # Bx512
        if self.args.swin == True:
            batch_size, h, w, c = aug2.shape
            aug2 = aug2.view(batch_size, h * w, c)


        # Project the image embeddings to the SSL embedding space
        aug1_embed = self.image_mlp(aug1) # Bx256
        aug2_embed = self.image_mlp(aug2) # Bx256
        
        # Get the logits for the SSL loss
        logits_ssl, labels_ssl = self.info_nce_loss(aug1_embed, aug2_embed)
        # ------------------------------ Image SSL branch --------------------------------

        # ------------------------------ Image-Text dot product branch --------------------------------
        # Split the prompts into 2 views
        text_embedding_v1 = []
        text_embedding_v2 = []
        for label in source_labels:
            label = int(label.item())

            if label == 0: # spoof
                # Randomly choose indices for the 2 views
                available_indices = np.arange(0, len(spoof_templates))
                pair_1 = np.random.choice(available_indices, len(spoof_templates)//2)
                pair_2 = np.setdiff1d(available_indices, pair_1)
                # slice embedding based on the indices
                spoof_texts_v1 = [ all_spoof_class_embeddings[i] for i in pair_1] # slice from the embedded fake templates
                spoof_texts_v2 = [ all_spoof_class_embeddings[i] for i in pair_2] # slice from the embedded fake templates
                # stack the embeddings
                spoof_texts_v1 = torch.stack(spoof_texts_v1, dim=0).cuda() # 3x512
                spoof_texts_v2 = torch.stack(spoof_texts_v2, dim=0).cuda() # 3x512
                assert int(spoof_texts_v1.shape[1]) == 512 and int(spoof_texts_v2.shape[1]) == 512 , "text embedding shape is not 512"
                # append the embeddings
                text_embedding_v1.append(spoof_texts_v1.mean(dim=0))
                text_embedding_v2.append(spoof_texts_v2.mean(dim=0))

            elif label == 1: # real
                # Randomly choose indices for the 2 views
                available_indices = np.arange(0, len(real_templates))
                pair_1 = np.random.choice(available_indices, len(real_templates)//2)
                pair_2 = np.setdiff1d(available_indices, pair_1)
                # slice embedding based on the indices
                real_texts_v1 = [ all_real_class_embeddings[i] for i in pair_1] # slice from the tokenized templates
                real_texts_v2 = [ all_real_class_embeddings[i] for i in pair_2] # slice from the tokenized templates
                # stack the embeddings
                real_texts_v1 = torch.stack(real_texts_v1, dim=0).cuda() # 3x512
                real_texts_v2 = torch.stack(real_texts_v2, dim=0).cuda() # 3x512
                assert int(real_texts_v1.shape[1]) == 512 and int(real_texts_v2.shape[1]) == 512 , "text embedding shape is not 512"
                # append the embeddings
                text_embedding_v1.append(real_texts_v1.mean(dim=0))
                text_embedding_v2.append(real_texts_v2.mean(dim=0))
    
        text_embed_v1 = torch.stack(text_embedding_v1, dim=0).cuda() # Bx512
        text_embed_v2 = torch.stack(text_embedding_v2, dim=0).cuda() # Bx512
        assert int(text_embed_v1.shape[1]) == 512 and int(text_embed_v2.shape[1]) == 512 , "text embedding shape is not 512"

        # dot product of image and text embeddings
        aug1_norm = aug1 / aug1.norm(dim=-1, keepdim=True)
        aug2_norm = aug2 / aug2.norm(dim=-1, keepdim=True)

        text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim=-1, keepdim=True)
        text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim=-1, keepdim=True)

        aug1_text_dot_product = self.cosine_similarity(aug1_norm, text_embed_v1_norm)
        aug2_text_dot_product = self.cosine_similarity(aug2_norm, text_embed_v2_norm)

        # mse loss between the dot product of aug1 and aug2
        dot_product_loss = self.mse_loss(aug1_text_dot_product, aug2_text_dot_product)
        # ------------------------------ Image-Text dot product branch --------------------------------


        #--------------------------------Knowledge distillation loss-----------------------------------
        if self.args.t_embed_dim != self.args.s_embed_dim:
            print("t_embed_dim != s_embed_dim")
            image_features = self.visual_proj(image_features)
            text_features = self.text_proj(text_features)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) 

        fd_loss = torch.tensor(0.).cuda()
        if self.args.alpha_fd_loss > 0:
            #fd_loss = F.mse_loss(image_features, t_image_features) + F.mse_loss(text_features, t_text_features)
            fd_loss = F.mse_loss(image_features, t_image_features)
            fd_loss = self.args.alpha_fd_loss * fd_loss

        ckd_loss = torch.tensor(0.).cuda()
        if self.args.alpha_ckd_loss > 0:
            ckd_loss = (self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
                        self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2
            ckd_loss = self.args.alpha_ckd_loss * ckd_loss

        affinity_loss = torch.tensor(0.).cuda()
        if self.args.alpha_affinity_loss > 0:
            affinity_loss = (F.cross_entropy(logits_per_image, t_logits_per_image.detach()) +\
                            F.cross_entropy(logits_per_text, t_logits_per_text.detach())) / 2
            affinity_loss = self.args.alpha_affinity_loss * affinity_loss

        gd_loss = torch.tensor(0.).cuda()
        if self.args.alpha_gd_loss > 0.:
            for label in source_labels:
                #label = int(label.item())

                with torch.no_grad():
                    t_grad_p_img, t_grad_k_txt = self.get_grad(t_image_features, t_text_features, t_logit_scale, label)
                    t_grad_p_txt, t_grad_k_img = self.get_grad(t_text_features, t_image_features, t_logit_scale, label)
            
                s_grad_p_img, s_grad_k_txt = self.get_grad(image_features, text_features, logit_scale, label)
                s_grad_p_txt, s_grad_k_img = self.get_grad(text_features, image_features, logit_scale, label)
        
                gd_loss += F.mse_loss(s_grad_p_img, t_grad_p_img.detach()) +\
                    F.mse_loss(s_grad_k_txt, t_grad_k_txt.detach()) +\
                    F.mse_loss(s_grad_p_txt, t_grad_p_txt.detach()) +\
                    F.mse_loss(s_grad_k_img, t_grad_k_img.detach()) 
                
            gd_loss = self.args.alpha_gd_loss * gd_loss

        #------------------------------------------RKD Loss---------------------------------------------------
        
        triplet_loss = torch.tensor(0.).cuda()
        if self.args.triplet_ratio > 0 :
            triplet_criterion = L2Triplet(sampler=self.args.triplet_sample(), margin=self.args.triplet_margin)
            triplet_loss = self.args.triplet_ratio * triplet_criterion(image_features, source_labels)
        
        dist_loss = torch.tensor(0.).cuda()
        if self.args.dist_ratio > 0 :
            dist_criterion = RkdDistance()
            dist_loss = (dist_criterion(image_features, t_image_features) + dist_criterion(text_features, t_text_features)) / 2
            dist_loss = self.args.dist_ratio * dist_loss

        angle_loss = torch.tensor(0.).cuda()
        if self.args.angle_ratio > 0 :
            angle_criterion = RKdAngle()
            angle_loss = (angle_criterion(image_features, t_image_features) + angle_criterion(text_features, t_text_features)) / 2
            angle_loss = self.args.angle_ratio * angle_loss

        dark_loss = torch.tensor(0.).cuda()
        if self.args.dark_ratio > 0 :
            dark_criterion = HardDarkRank(alpha=self.args.dark_alpha, beta=self.args.dark_beta)
            dark_loss = (dark_criterion(image_features, t_image_features) + dark_criterion(text_features, t_text_features))/2
            dark_loss = self.args.dark_ratio * dark_loss

        at_loss = torch.tensor(0.).cuda()
        if self.args.at_ratio > 0 :
            at_criterion = AttentionTransfer()
            #at_loss = self.args.at_ratio * (at_criterion(b2, t_b2) + at_criterion(b3, t_b3) + at_criterion(b4, t_b4))

        rkd_loss = triplet_loss + dist_loss + angle_loss + dark_loss + at_loss

        #----------------------------------------------------------------------------------------------

        distkd_loss = torch.tensor(0.).cuda()
        if self.args.distkd_ratio > 0 :
            kd_loss = DIST(beta=1, gamma=1, tau=1)
            distkd_loss = (kd_loss(logits_per_image, t_logits_per_image.detach()) + kd_loss(logits_per_text, t_logits_per_text.detach())) / 2
            distkd_loss = self.args.distkd_ratio * distkd_loss

        #------------------------------------------attention distillation------------------------------------------------

        attn_loss = torch.tensor(0.).cuda()
        if self.args.attn_ratio > 0:
            for i in range(12):
                s_attn = attention_weight[i][:, 1:, 1:]
                t_attn = t_attention_weight[i][:, 1:, 1:]
                attn_loss += F.mse_loss(s_attn, t_attn)

            attn_loss *= self.args.attn_ratio

        #----------------------------------------------------------------------------------------------------------------


        return similarity, logits_ssl, labels_ssl, dot_product_loss, fd_loss, ckd_loss, affinity_loss, gd_loss, rkd_loss, distkd_loss, attn_loss

    def forward_eval(self, input, norm_flag=True):
        # single text prompt per class
        # logits_per_image, logits_per_text = self.model(input, self.text_inputs)

        if self.args.model == 'ViT-B-16':
            self.model = self.load_tmodel()

        # Ensemble of text features
        ensemble_weights = []
        for classname in (['spoof', 'real']):
            if classname == 'spoof':
                texts = spoof_templates #format with spoof class
            elif classname == 'real':
                texts = real_templates #format with real class
            
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = self.model.encode_text(texts) #embed with text encoder
            class_embedding = class_embeddings.mean(dim=0)
            ensemble_weights.append(class_embedding)
        text_features = torch.stack(ensemble_weights, dim=0).cuda()

        # get the image features
        if self.args.vis == True:
            image_features, attention_map = self.model.encode_image(input)
        else:
            image_features, _ = self.model.encode_image(input)
        if self.args.swin == True:
            batch_size, h, w, c = image_features.shape
            image_features = image_features.view(batch_size, h * w, c)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        similarity = logits_per_image

        if self.args.vis == True:
            return similarity, attention_map
        else:
            return similarity
    
    def forward_tsne(self, input, norm_flag=True):
        # single text prompt per class
        # logits_per_image, logits_per_text = self.model(input, self.text_inputs)

        # Ensemble of text features
        if self.args.model == 'ViT-B-16':
            self.model = self.t_model
        ensemble_weights = []
        for classname in (['spoof', 'real']):
            if classname == 'spoof':
                texts = spoof_templates #format with spoof class
            elif classname == 'real':
                texts = real_templates #format with real class
            
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = self.model.encode_text(texts) #embed with text encoder
            class_embedding = class_embeddings.mean(dim=0)
            ensemble_weights.append(class_embedding)
        text_features = torch.stack(ensemble_weights, dim=0).cuda()

        # get the image features
        image_features, _ = self.model.encode_image(input)

        if self.args.swin == True:
            batch_size, h, w, c = image_features.shape
            image_features = image_features.view(batch_size, h * w, c)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        similarity = logits_per_image
        return image_features, text_features


    def forward_vis(self, input, norm_flag=True):
        # image_features, image_features_proj = self.model.encode_image(input)
        # _, image_features_proj = self.model.visual.forward_full(input)
        image_features, image_features_proj = self.model.visual.forward_full(input)
        feature = image_features_proj

        # return None, feature
        return image_features, feature
