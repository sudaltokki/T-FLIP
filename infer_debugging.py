import sys

# sys.path.append('../../')

from utils.evaluate import eval
from utils.dataset import get_dataset_ssl_clip
from train.fas import flip_mcl
import numpy as np
from train.config import configC, configM, configI, configO

import time
from timeit import default_timer as timer
import os
import torch
import argparse
from train.params import parse_args
import json
from PIL import Image


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


def infer(args, config):
    _, _, _, _, _, _, _, _, _, _, test_dataloader = get_dataset_ssl_clip(args,   
        config.src1_data, config.src1_train_num_frames,
        config.src2_data, config.src2_train_num_frames,
        config.src3_data, config.src3_train_num_frames,
        config.src4_data, config.src4_train_num_frames,
        config.src5_data, config.src5_train_num_frames,
        config.tgt_data, config.tgt_test_num_frames)

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    best_TPR_FPR = 0.0
    
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]
    
    net1 = flip_mcl(args, device, in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device)

    if config.checkpoint:
        ckpt = torch.load(config.checkpoint)
        net1.load_state_dict(ckpt['state_dict'])
        epoch = ckpt['epoch']
        iter_num_start = epoch*100
        print(f'Loaded checkpoint from epoch {epoch} at iteration : {iter_num_start}' )


    ######### eval #########
    valid_args = eval(test_dataloader, net1, True, vis=args.vis)
    # judge model according to HTER
    is_best = valid_args[3] <= best_model_HTER
    best_model_HTER = min(valid_args[3], best_model_HTER)
    threshold = valid_args[5]

    best_model_ACC = valid_args[6]
    best_model_AUC = valid_args[4]
    best_TPR_FPR = valid_args[-2]
            

    return best_model_HTER*100.0, best_model_AUC*100.0, best_TPR_FPR*100.0, valid_args[8], valid_args[9]


def main(args):

    args = parse_args(args)

    with open(os.path.join(os.getcwd(), 'student/model_config/'+args.t_model+'.json'), 'r') as f:
        args.t_embed_dim = json.load(f)['embed_dim']
    with open(os.path.join(os.getcwd(), 'student/model_config/'+args.model+'.json'), 'r') as f:
        args.s_embed_dim = json.load(f)['embed_dim']

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            args.current_time.strftime("%Y_%m_%d-%H_%M_%S"),
            f"t_model_{args.t_model}",
            f"s_model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}"
        ])

    log_base_path = os.path.join(args.report_logger_path, args.name)
    os.makedirs(log_base_path, exist_ok = True)

  # 0-shot / 5-shot
    if args.config == 'I':
        config = configI
    if args.config == 'C':
        config = configC
    if args.config == 'M':
        config = configM
    if args.config == 'O':
        config = configO

    for attr in dir(config):
        if attr.find('__') == -1:
            print('%s = %r' % (attr, getattr(config, attr)))
  
    with open(os.path.join(args.report_logger_path, args.name, 'out.log'), "w") as f:
        f.write('HTER, AUC, TPR@FPR=1%\n')

        config.checkpoint = args.ckpt
        hter, auc, tpr_fpr, true_false_list, attention_map = infer(args, config)

        f.write(f'{hter},{auc},{tpr_fpr}\n')

    output_directory = "debugging_images"
    output_directory = os.path.join(args.report_logger_path, args.name, output_directory)
    os.makedirs(output_directory, exist_ok=True)

    for image_path in true_false_list:
        try:
            img = Image.open(image_path)
            image_filename = os.path.basename(image_path)
            
            output_path = os.path.join(output_directory, image_filename)
            img.save(output_path)
            
        except Exception as e:
            print(f"Failed to save {image_path}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
