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
from third_party.utils.random import random_seed
from student.fas import flip_mcl
import json
import wandb


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)


def infer(config, args):

  args_info = "\n".join([f"{arg}: {getattr(args, arg)}" for arg in vars(args)])
  logging.info(f"\n-----------------------------Arguments----------------------------|\n{args_info}\n")
  
  _, _, _, _, _, _, _, _, _, _, test_dataloader = get_dataset_ssl_clip(args,  
      config.src1_data, config.src1_train_num_frames, config.src2_data,
      config.src2_train_num_frames, config.src3_data,
      config.src3_train_num_frames, config.src4_data,
      config.src4_train_num_frames, config.src5_data,
      config.src5_train_num_frames, config.tgt_data, tgt_test_num_frames=config.tgt_test_num_frames,collate_fn=custom_collate_fn)

  best_model_ACC = 0.0
  best_model_HTER = 1.0
  best_model_ACER = 1.0
  best_model_AUC = 0.0
  best_TPR_FPR = 0.0
  
  valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]
  
  ###################################################
  random_seed(42, 0)
  t_model, _ = clip.load("ViT-B/16", 'cuda:0')
  model = flip_mcl(args, device, in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device)


  ckpt = torch.load(config.checkpoint)
  model.load_state_dict(ckpt['state_dict'])
  model.eval()


  ######### eval #########
  valid_args = eval(test_dataloader, model, True)
  # judge model according to HTER
  is_best = valid_args[3] <= best_model_HTER
  best_model_HTER = min(valid_args[3], best_model_HTER)
  threshold = valid_args[5]

  best_model_ACC = valid_args[6]
  best_model_AUC = valid_args[4]
  best_TPR_FPR = valid_args[-1]
        

  return best_model_HTER*100.0, best_model_AUC*100.0, best_TPR_FPR*100.0



if __name__ == '__main__':
  args = sys.argv[1:]
  args = parse_args(args)

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
    f.write('Run, HTER, AUC, TPR@FPR=1%\n')
    hter_avg = []
    auc_avg = []
    tpr_fpr_avg = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(1):
      # To reproduce results
      torch.manual_seed(i)
      np.random.seed(i)

      config.run = i
      config.checkpoint = args.ckpt
      # start_time = time.process_time()
      start_event.record()
      hter, auc, tpr_fpr = infer(config,args)
      # end_time = time.process_time()
      end_event.record()

      torch.cuda.synchronize()

      hter_avg.append(hter)
      auc_avg.append(auc)
      tpr_fpr_avg.append(tpr_fpr)

      f.write(f'{i},{hter},{auc},{tpr_fpr}\n')

    
    hter_mean = np.mean(hter_avg)
    auc_mean = np.mean(auc_avg)
    tpr_fpr_mean = np.mean(tpr_fpr_avg)
    f.write(f'Mean,{hter_mean},{auc_mean},{tpr_fpr_mean}\n')

    hter_std = np.std(hter_avg)
    auc_std = np.std(auc_avg)
    tpr_fpr_std = np.std(tpr_fpr_avg)
    f.write(f'Std dev,{hter_std},{auc_std},{tpr_fpr_std}\n')

    f.write(f'\n')

    # elapsed_time = end_time - start_time
    elapsed_time = start_event.elapsed_time(end_event)
    f.write(f'GPU time used,{elapsed_time}\n')

    ###################################################
    # f.write()
    ###################################################
