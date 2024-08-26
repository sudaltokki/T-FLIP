import sys

# sys.path.append('../../')

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset
from utils.dataset import get_dataset_one_to_one_ssl_clip , get_dataset_ssl_clip
from student.fas import flip_mcl, flip_v, flip_it
import random
import numpy as np
from teacher.config import configC, configM, configI, configO, config_cefa, config_surf, config_wmca
from teacher.config import config_CI, config_CO , config_CM, config_MC, config_MI, config_MO, config_IC, config_IO, config_IM, config_OC, config_OI, config_OM
from datetime import datetime
import time
from timeit import default_timer as timer
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from student.params import parse_args
import clip
from clip.model import CLIP
import logging
from utils.logger import setup_logging
from student.fas import flip_mcl
import json
import wandb
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve
import os

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def get_EER_states(probabilities, labels):
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return eer, eer_threshold, fpr, tpr

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

def eval_with_analysis(valid_dataloader, model, norm_flag, return_prob=False):
    """
    FAS 모델을 대상 데이터에서 평가합니다.
    평가 중 잘 예측된 데이터와 잘못 예측된 데이터를 분리하고 분석합니다.
    """

    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    incorrect_filenames = []  # 잘못 예측된 데이터 파일명을 저장할 리스트

    with torch.no_grad():
        for iter, (input, target, videoID, name) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

            cls_out = model.forward_eval(input, norm_flag)

            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()

            for i in range(len(prob)):
                if videoID[i] in prob_dict.keys():
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = [prob[i]]
                    label_dict[videoID[i]] = [label[i]]
                    output_dict_tmp[videoID[i]] = [cls_out[i].view(1, 2)]
                    target_dict_tmp[videoID[i]] = [target[i].view(1)]

            # 잘못 예측된 파일명을 기록합니다.
            pred_label = np.argmax(prob)
            if pred_label != label[i]:
                incorrect_filenames.append(name[i])

    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)

        # 각 비디오에 대한 loss와 accuracy 계산
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        loss = criterion(avg_single_video_output, avg_single_video_target.long())
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])

    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    fpr, tpr, thr = roc_curve(label_list, prob_list)
    tpr_filtered = tpr[fpr <= 1 / 100]
    rate = tpr_filtered[-1] if len(tpr_filtered) > 0 else 0
    print("TPR@FPR = ", rate)

    # 잘못된 파일명을 txt 파일로 저장
    with open('incorrect_filenames.txt', 'w') as f:
        for filename in incorrect_filenames:
            f.write(f"{filename}\n")

    if not return_prob:
        return [
            valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
            auc_score, threshold, ACC_threshold * 100, rate
        ]
    else:
        return [
            valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
            auc_score, threshold, ACC_threshold * 100, rate
        ], [prob_list, label_list]

def analyze_samples(correct_samples, incorrect_samples):
    # t-SNE 시각화 - 정확하게 예측된 데이터
    correct_embeddings = np.array([sample['embedding'].numpy().flatten() for sample in correct_samples])
    correct_tsne = TSNE(n_components=2, random_state=42)
    correct_tsne_results = correct_tsne.fit_transform(correct_embeddings)

    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(correct_tsne_results):
        color = 'green' if correct_samples[i]['target'] == 0 else 'orange'
        plt.scatter(x, y, color=color, label=f"Pred: {correct_samples[i]['predicted']}, True: {correct_samples[i]['target']}")
    plt.title("t-SNE of Correct Predictions")
    plt.legend(loc='best')
    plt.show()

    # t-SNE 시각화 - 잘못 예측된 데이터
    incorrect_embeddings = np.array([sample['embedding'].numpy().flatten() for sample in incorrect_samples])
    incorrect_tsne = TSNE(n_components=2, random_state=42)
    incorrect_tsne_results = incorrect_tsne.fit_transform(incorrect_embeddings)

    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(incorrect_tsne_results):
        color = 'red' if incorrect_samples[i]['target'] == 0 else 'blue'
        plt.scatter(x, y, color=color, label=f"Pred: {incorrect_samples[i]['predicted']}, True: {incorrect_samples[i]['target']}")
    plt.title("t-SNE of Incorrect Predictions")
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':
  args = sys.argv[1:]
  args = parse_args(args)

  random_seed()

  with open(os.path.join(os.getcwd(), 'student/model_config/'+args.t_model+'.json'), 'r') as f:
        args.t_embed_dim = json.load(f)['embed_dim']
  with open(os.path.join(os.getcwd(), 'student/model_config/'+args.model+'.json'), 'r') as f:
      args.s_embed_dim = json.load(f)['embed_dim']


  # 0-shot / 5-shot
  if args.config == 'I':
    config = configI
  if args.config == 'C':
    config = configC
  if args.config == 'M':
    config = configM
  if args.config == 'O':
    config = configO
  if args.config == 'cefa':
    config = config_cefa
  if args.config == 'surf':
    config = config_surf
  if args.config == 'wmca':
    config = config_wmca


  for attr in dir(config):
    if attr.find('__') == -1:
      print('%s = %r' % (attr, getattr(config, attr)))

  config.checkpoint = args.ckpt
  
  with open(args.report_logger_path, "w") as f:

    config.checkpoint = args.ckpt

    _, _, _, _, _, _, _, _, _, _, valid_dataloader = get_dataset_ssl_clip(args,  
        config.src1_data, config.src1_train_num_frames, config.src2_data,
        config.src2_train_num_frames, config.src3_data,
        config.src3_train_num_frames, config.src4_data,
        config.src4_train_num_frames, config.src5_data,
        config.src5_train_num_frames, config.tgt_data, tgt_test_num_frames=config.tgt_test_num_frames, collate_fn=custom_collate_fn)

    model = flip_mcl(args, device, in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device)

    ckpt = torch.load(config.checkpoint)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    norm_flag = True

    results = eval_with_analysis(valid_dataloader, model, norm_flag)

    
    # 잘못된 예측에 대한 추가 분석을 위해 아래와 같이 실행할 수 있습니다.
    results, prob_label_lists = eval_with_analysis(valid_dataloader, model, norm_flag, return_prob=True)


