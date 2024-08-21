import logging
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import wandb

from timeit import default_timer as timer

from utils.utils import save_checkpoint, AverageMeter, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset_one_to_one_ssl_clip , get_dataset_ssl_clip

from student.fas import flip_mcl


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


def train(config, args):
  
    # # # 5-shot
    # # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, _, _, src5_train_dataloader_fake, src5_train_dataloader_real, test_dataloader = get_dataset_ssl_clip( # for wcs
    # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, src4_train_dataloader_fake, src4_train_dataloader_real, src5_train_dataloader_fake, src5_train_dataloader_real, test_dataloader = get_dataset_ssl_clip( # for mcio
    #     config.src1_data, config.src1_train_num_frames, config.src2_data,
    #     config.src2_train_num_frames, config.src3_data,
    #     config.src3_train_num_frames, config.src4_data,
    #     config.src4_train_num_frames, config.src5_data,
    #     config.src5_train_num_frames, config.tgt_data, config.tgt_test_num_frames)

    # 0-shot
    # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, _, _, _, _, test_dataloader = get_dataset_ssl_clip( # for wcs
    src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, src4_train_dataloader_fake, src4_train_dataloader_real, _, _, test_dataloader = get_dataset_ssl_clip( # for mcio
        args, config.src1_data, config.src1_train_num_frames, config.src2_data,
        config.src2_train_num_frames, config.src3_data,
        config.src3_train_num_frames, config.src4_data,
        config.src4_train_num_frames, config.src5_data,
        config.src5_train_num_frames, config.tgt_data, config.tgt_test_num_frames)

    # 1-1 setting
    # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, _, _, test_dataloader = get_dataset_one_to_one_ssl_clip( # for 0-shot in 1-1 setting
    # src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, test_dataloader = get_dataset_one_to_one_ssl_clip( # for 5-shot in 1-1 setting.
        # config.src1_data, config.src1_train_num_frames, 
        # config.src2_data, config.src2_train_num_frames,
        # config.src3_data, config.src3_train_num_frames,
        # config.tgt_data, config.tgt_test_num_frames)

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    best_TPR_FPR = 0.0

    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_simclr = AverageMeter()
    loss_l2_euclid = AverageMeter()
    loss_total = AverageMeter()
    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()
    loss_fd = AverageMeter()
    loss_ckd = AverageMeter()
    loss_affinity = AverageMeter()
    loss_icl = AverageMeter()
    loss_rkd = AverageMeter()

    logging.info('\n----------------------------------------------- [START %s] %s' %
        (args.current_time.strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    logging.info('** start training target model! **')
    logging.info('--------|------------- VALID -------------|--- classifier ---|-----------------SimCLR loss-------------|-----------------KD loss--------------|----- RKD loss -----|-------- Current Best --------|--------------|')
    logging.info('  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |  SimCLR-loss    l2-loss    total-loss   |    fd_loss    ckd_loss    affinity   |      rkd_loss      |    top-1    HTER     AUC     |     time     |')
    logging.info('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|')
    
    start = timer()
    criterion = {'softmax': nn.CrossEntropyLoss().cuda()}
    
    model = flip_mcl(args, device, in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device) # ssl applied to image, and euclidean distance applied to image and text cosine similarity

    # Fine-tune all the layers
    for name, param in model.named_parameters():
        param.requires_grad = True

    # Load if checkpoint is provided
    if config.checkpoint:
        ckpt = torch.load(config.checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        epoch = ckpt['epoch']
        iter_num_start = epoch*100
        logging.info(f'Loaded checkpoint from epoch {epoch} at iteration : {iter_num_start}' )
    else:
        epoch = 1
        iter_num_start = 0
        logging.info(f'Starting training from epoch {epoch} at iteration : {iter_num_start}' )
        
    iteration = args.iterations
    iter_per_epoch = args.epochs

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src4_train_iter_real = iter(src4_train_dataloader_real)
    src4_iter_per_epoch_real = len(src4_train_iter_real)
    # comment the following 2 lines when training in 0-shot
    # src5_train_iter_real = iter(src5_train_dataloader_real)
    # src5_iter_per_epoch_real = len(src5_train_iter_real)

    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)
    src4_train_iter_fake = iter(src4_train_dataloader_fake)
    src4_iter_per_epoch_fake = len(src4_train_iter_fake)
    # comment the following 2 lines when training in 0-shot
    # src5_train_iter_fake = iter(src5_train_dataloader_fake)
    # src5_iter_per_epoch_fake = len(src5_train_iter_fake)

    optimizer_dict = [
        {
            'params': filter(lambda p: p.requires_grad, model.parameters()),
            'lr': args.lr,
            'weight_decay': args.wd
        },
    ]

    optimizer = optim.AdamW(optimizer_dict)

    total_steps = min(
    src1_iter_per_epoch_real, src1_iter_per_epoch_fake,
    src2_iter_per_epoch_real, src2_iter_per_epoch_fake,
    src3_iter_per_epoch_real, src3_iter_per_epoch_fake,
    src4_iter_per_epoch_real, src4_iter_per_epoch_fake
    )
    total_steps = total_steps * iteration

    scheduler = CosineAnnealingLR(optimizer, total_steps)

  
    for iter_num in range(iter_num_start, iteration + 1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if (iter_num % src4_iter_per_epoch_real == 0):
            src4_train_iter_real = iter(src4_train_dataloader_real)
        # comment the following 2 lines when training in 0-shot
        # if (iter_num % src5_iter_per_epoch_real == 0):
        #       src5_train_iter_real = iter(src5_train_dataloader_real)

        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if (iter_num % src4_iter_per_epoch_fake == 0):
            src4_train_iter_fake = iter(src4_train_dataloader_fake)
        # comment the following 2 lines when training in 0-shot
        # if (iter_num % src5_iter_per_epoch_fake == 0):
        #   src5_train_iter_fake = iter(src5_train_dataloader_fake)

        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1

        # net1.train(True)
        optimizer.zero_grad()

        ######### data prepare #########
        src1_img_real, src1_img_real_view_1, src1_img_real_view_2, src1_label_real = src1_train_iter_real.__next__()
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        src1_img_real_view_1 = src1_img_real_view_1.cuda()
        src1_img_real_view_2 = src1_img_real_view_2.cuda()
        input1_real_shape = src1_img_real.shape[0]

        src2_img_real, src2_img_real_view_1, src2_img_real_view_2, src2_label_real = src2_train_iter_real.__next__()
        src2_img_real = src2_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        src2_img_real_view_1 = src2_img_real_view_1.cuda()
        src2_img_real_view_2 = src2_img_real_view_2.cuda()
        input2_real_shape = src2_img_real.shape[0]

        src3_img_real, src3_img_real_view_1, src3_img_real_view_2, src3_label_real = src3_train_iter_real.__next__()
        src3_img_real = src3_img_real.cuda()
        src3_label_real = src3_label_real.cuda()
        src3_img_real_view_1 = src3_img_real_view_1.cuda()
        src3_img_real_view_2 = src3_img_real_view_2.cuda()
        input3_real_shape = src3_img_real.shape[0]

        src4_img_real, src4_img_real_view_1, src4_img_real_view_2, src4_label_real = src4_train_iter_real.__next__()
        src4_img_real = src4_img_real.cuda()
        src4_label_real = src4_label_real.cuda()
        src4_img_real_view_1 = src4_img_real_view_1.cuda()
        src4_img_real_view_2 = src4_img_real_view_2.cuda()
        input4_real_shape = src4_img_real.shape[0]

        # comment the following 6 lines when training in 0-shot
        # src5_img_real, src5_img_real_view_1, src5_img_real_view_2, src5_label_real = src5_train_iter_real.next()
        # src5_img_real = src5_img_real.cuda()
        # src5_label_real = src5_label_real.cuda()
        # src5_img_real_view_1 = src5_img_real_view_1.cuda()
        # src5_img_real_view_2 = src5_img_real_view_2.cuda()
        # input5_real_shape = src5_img_real.shape[0]

        src1_img_fake, src1_img_fake_view_1, src1_img_fake_view_2, src1_label_fake = src1_train_iter_fake.__next__()
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        src1_img_fake_view_1 = src1_img_fake_view_1.cuda()
        src1_img_fake_view_2 = src1_img_fake_view_2.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        src2_img_fake, src2_img_fake_view_1, src2_img_fake_view_2, src2_label_fake = src2_train_iter_fake.__next__()
        src2_img_fake = src2_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
        src2_img_fake_view_1 = src2_img_fake_view_1.cuda()
        src2_img_fake_view_2 = src2_img_fake_view_2.cuda()
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_fake, src3_img_fake_view_1, src3_img_fake_view_2, src3_label_fake = src3_train_iter_fake.__next__()
        src3_img_fake = src3_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
        src3_img_fake_view_1 = src3_img_fake_view_1.cuda()
        src3_img_fake_view_2 = src3_img_fake_view_2.cuda()
        input3_fake_shape = src3_img_fake.shape[0]

        src4_img_fake, src4_img_fake_view_1, src4_img_fake_view_2, src4_label_fake = src4_train_iter_fake.__next__()
        src4_img_fake = src4_img_fake.cuda()
        src4_label_fake = src4_label_fake.cuda()
        src4_img_fake_view_1 = src4_img_fake_view_1.cuda()
        src4_img_fake_view_2 = src4_img_fake_view_2.cuda()
        input4_fake_shape = src4_img_fake.shape[0]

        # comment the following 6 lines when training in 0-shot
        # src5_img_fake, src5_img_fake_view_1, src5_img_fake_view_2, src5_label_fake = src5_train_iter_fake.next()
        # src5_img_fake = src5_img_fake.cuda()
        # src5_label_fake = src5_label_fake.cuda()
        # src5_img_fake_view_1 = src5_img_fake_view_1.cuda()
        # src5_img_fake_view_2 = src5_img_fake_view_2.cuda()
        # input5_fake_shape = src5_img_fake.shape[0]
    

        if config.tgt_data in ['cefa', 'surf', 'wmca']:
            input_data = torch.cat([
                src1_img_real, src1_img_fake, 
                src2_img_real, src2_img_fake,
                src3_img_real, src3_img_fake, 
                # src5_img_real, src5_img_fake
            ],
                                    dim=0)

            input_data_view_1 = torch.cat([
                src1_img_real_view_1, src1_img_fake_view_1, 
                src2_img_real_view_1, src2_img_fake_view_1,
                src3_img_real_view_1, src3_img_fake_view_1,
                src4_img_real_view_1, src4_img_fake_view_1,
                # src5_img_real_view_1, src5_img_fake_view_1
            ], dim=0)

            input_data_view_2 = torch.cat([
                src1_img_real_view_2, src1_img_fake_view_2, 
                src2_img_real_view_2, src2_img_fake_view_2,
                src3_img_real_view_2, src3_img_fake_view_2,
                src4_img_real_view_2, src4_img_fake_view_2,
                # src5_img_real_view_2, src5_img_fake_view_2
            ], dim=0)

        else:
            input_data = torch.cat([
                src1_img_real, src1_img_fake, 
                src2_img_real, src2_img_fake,
                src3_img_real, src3_img_fake, 
                src4_img_real, src4_img_fake,
                # src5_img_real, src5_img_fake
            ],
                                    dim=0)

            input_data_view_1 = torch.cat([
                src1_img_real_view_1, src1_img_fake_view_1, 
                src2_img_real_view_1, src2_img_fake_view_1,
                src3_img_real_view_1, src3_img_fake_view_1,
                src4_img_real_view_1, src4_img_fake_view_1,
                # src5_img_real_view_1, src5_img_fake_view_1
            ], dim=0)

            input_data_view_2 = torch.cat([
                src1_img_real_view_2, src1_img_fake_view_2, 
                src2_img_real_view_2, src2_img_fake_view_2,
                src3_img_real_view_2, src3_img_fake_view_2,
                src4_img_real_view_2, src4_img_fake_view_2,
                # src5_img_real_view_2, src5_img_fake_view_2
            ], dim=0)

        if config.tgt_data in ['cefa', 'surf', 'wmca']:
            source_label = torch.cat([
                src1_label_real.fill_(1),
                src1_label_fake.fill_(0),
                src2_label_real.fill_(1),
                src2_label_fake.fill_(0),
                src3_label_real.fill_(1),
                src3_label_fake.fill_(0),
                # src5_label_real.fill_(1),
                # src5_label_fake.fill_(0)
            ],
                                    dim=0)
        
        else:
            source_label = torch.cat([
                src1_label_real.fill_(1),
                src1_label_fake.fill_(0),
                src2_label_real.fill_(1),
                src2_label_fake.fill_(0),
                src3_label_real.fill_(1),
                src3_label_fake.fill_(0),
                src4_label_real.fill_(1),
                src4_label_fake.fill_(0),
                # src5_label_real.fill_(1),
                # src5_label_fake.fill_(0)
            ],
                                    dim=0)

        ######### forward #########

        #gradient accumulation
        accumulation_step = int(args.total_batch_size / args.batch_size)

        # output image feature + text feature
        classifier_label_out , logits_ssl, labels_ssl, l2_euclid_loss, fd_loss, ckd_loss, affinity_loss, icl_loss, rkd_loss = model(input_data, input_data_view_1, input_data_view_2, source_label, True) # ce on I-T, SSL for image and l2 loss for image-view-text dot product
        cls_loss = criterion['softmax'](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)
        sim_loss = criterion['softmax'](logits_ssl, labels_ssl) 

        fac = 1.0
        total_loss = cls_loss + fac*sim_loss + fac*l2_euclid_loss + fd_loss + ckd_loss + affinity_loss + icl_loss + rkd_loss

        total_loss.backward()
        if (iter_num+1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        loss_classifier.update(cls_loss.item())
        loss_l2_euclid.update(l2_euclid_loss.item())
        loss_simclr.update(sim_loss.item())
        loss_total.update(total_loss.item())
        loss_fd.update(fd_loss.item())
        loss_ckd.update(ckd_loss.item())
        loss_affinity.update(affinity_loss.item())
        loss_icl.update(icl_loss.item())
        loss_rkd.update(rkd_loss.item())

        acc = accuracy(
            classifier_label_out.narrow(0, 0, input_data.size(0)),
            source_label,
            topk=(1,))
        classifer_top1.update(acc[0])


        if (iter_num != 0 and (iter_num + 1) % (iter_per_epoch) == 0):
            valid_args = eval(test_dataloader, model, True)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]
                best_TPR_FPR = valid_args[-1]

            save_list = [
            epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER,
            threshold
            ]

            save_checkpoint(save_list, is_best, model, args.op_dir, f'flip_mcl_checkpoint_run_{str(config.run)}.pth.tar')


            print('\r', end='', flush=True)
            logging.info(
                '  %4.1f |   %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |     %6.3f      %6.3f      %6.3f     |     %6.3f     %6.3f     %6.3f     |       %6.3f       |    %6.3f  %6.3f  %6.3f    | %s   %s'
                % ((iter_num + 1) / iter_per_epoch, 
                    valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100, 
                    loss_classifier.avg, classifer_top1.avg,
                    loss_simclr.avg, loss_l2_euclid.avg, loss_total.avg,
                    loss_fd.avg, loss_ckd.avg, loss_affinity.avg, loss_rkd.avg,
                    float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100), time_to_str(timer() - start, 'min'), 0))

            time.sleep(0.01)
            if args.set_wandb:
                wandb.log({'SimCLR-loss' : loss_simclr.avg, 'l2-loss' : loss_l2_euclid.avg, 'total-loss': loss_total.avg, 'fd_loss': loss_fd.avg, 'ckd_loss': loss_ckd.avg, 'affinity_loss':loss_affinity.avg, 'rkd_loss': loss_rkd.avg, 'valid_loss': valid_args[0], 'top-1':valid_args[6], 'HTER':valid_args[3] * 100, 'AUC':valid_args[4] * 100, 'classifier_loss':loss_classifier.avg, 'classifier_top1': classifer_top1.avg})

    return best_model_HTER*100.0, best_model_AUC*100.0, best_TPR_FPR*100.0
