import json
import math
import pandas as pd
import torch
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import wandb

def get_datetime():
    tz = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(tz)
    return current_time

def draw_roc(frr_list, far_list, roc_auc):
  plt.switch_backend('agg')
  plt.rcParams['figure.figsize'] = (6.0, 6.0)
  plt.title('ROC')
  plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
  plt.legend(loc='upper right')
  plt.plot([0, 1], [1, 0], 'r--')
  plt.grid(ls='--')
  plt.ylabel('False Negative Rate')
  plt.xlabel('False Positive Rate')
  save_dir = './save_results/ROC/'
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  plt.savefig('./save_results/ROC/ROC.png')
  file = open('./save_results/ROC/FAR_FRR.txt', 'w')
  save_json = []
  dict = {}
  dict['FAR'] = far_list
  dict['FRR'] = frr_list
  save_json.append(dict)
  json.dump(save_json, file, indent=4)


def sample_frames(args, flag, num_frames, dataset_name):
  """
        from every video (frames) to sample num_frames to test
        return: the choosen frames' path and label
    """

  root = args.root
  dataroot = args.dataroot

  if (flag == 0):  # select the fake images
    data = [[
        root + i.strip()
        for i in open(dataroot + dataset_name + '_fake_train.txt').readlines()
    ], []]
    print('train fake:', dataset_name, len(data[0]), len(data[1]))

  elif (flag == 1):  # select the real images
    data = [[],
            [
                root + i.strip() for i in open(dataroot + dataset_name +
                                               '_real_train.txt').readlines()
            ]]
    print('train real:', dataset_name, len(data[0]), len(data[1]))

  elif (flag == 2):  # select the fake images
    data = [[
        root + i.strip()
        for i in open(dataroot + dataset_name + '_fake_shot.txt').readlines()
    ][:5], []]
    data = [data[0] + data[0] + data[0] + data[0], []]
    print('train fake:', dataset_name, len(data[0]), len(data[1]))

  elif (flag == 3):  # select the real images
    data = [[],
            [
                root + i.strip() for i in open(dataroot + dataset_name +
                                               '_real_shot.txt').readlines()
            ][:5]]
    data = [[], data[1] + data[1] + data[1] + data[1]]
    print('train real:', dataset_name, len(data[0]), len(data[1]))

  else:
    data = [[
        root + i.strip()
        for i in open(dataroot + dataset_name + '_fake_test.txt').readlines()
    ] + [
        root + i.strip()
        for i in open(dataroot + dataset_name + '_fake_test.txt').readlines()
    ],
            [
                root + i.strip() for i in open(dataroot + dataset_name +
                                               '_real_test.txt').readlines()
            ] + [
                root + i.strip()
                for i in open(dataroot + dataset_name +
                              '_real_test.txt').readlines()
            ]]
    data = [list(set(data[0])), list(set(data[1]))]
    print('test:', dataset_name, len(data[0]), len(data[1]))

  return data


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mkdirs(checkpoint_path, best_model_path, logs):
  if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
  if not os.path.exists(best_model_path):
    os.makedirs(best_model_path)
  if not os.path.exists(logs):
    os.mkdir(logs)


def time_to_str(t, mode='min'):
  if mode == 'min':
    t = int(t) / 60
    hr = t // 60
    min = t % 60
    return '%2d hr %02d min' % (hr, min)
  elif mode == 'sec':
    t = int(t)
    min = t // 60
    sec = t % 60
    return '%2d min %02d sec' % (min, sec)
  else:
    raise NotImplementedError


class Logger(object):

  def __init__(self):
    self.terminal = sys.stdout
    self.file = None

  def open(self, file, mode=None):
    if mode is None:
      mode = 'w'
    self.file = open(file, mode)

  def write(self, message, is_terminal=1, is_file=0):
    if '\r' in message:
      is_file = 0
    if is_terminal == 1:
      self.terminal.write(message)
      self.terminal.flush()
    if is_file == 1:
      self.file.write(message)
      self.file.flush()

  def flush(self):
    # this flush method is needed for python 3 compatibility.
    # this handles the flush command by doing nothing.
    # you might want to specify some extra behavior here.
    pass


def save_checkpoint(save_list, is_best, model, op_dir, name, filename='_checkpoint.pth.tar'):
  epoch = save_list[0]
  valid_args = save_list[1]
  best_model_HTER = round(save_list[2], 5)
  best_model_ACC = save_list[3]
  best_model_ACER = save_list[4]
  threshold = save_list[5]

  state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'valid_arg': valid_args,
      'best_model_EER': best_model_HTER,
      'best_model_ACER': best_model_ACER,
      'best_model_ACC': best_model_ACC,
      'threshold': threshold
  }

  ckpt_dir = os.path.join(op_dir, name)

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  filename = os.path.join(ckpt_dir, filename)

  if is_best:
    torch.save(state, filename)


def zero_param_grad(params):
  for p in params:
    if p.grad is not None:
      p.grad.zero_()

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def set_wandb(args, i):
    wandb_config= {
        "t_model": args.t_model,
        "model": args.model,
        "t_model_checkpoint": args.t_model_checkpoint,
        "alpha_ckd_loss": args.alpha_ckd_loss,
        "alpha_fd_loss": args.alpha_fd_loss,
        "alpha_affinity_loss": args.alpha_affinity_loss,
        "alpha_icl_loss": args.alpha_icl_loss,
        "dist_ratio": args.dist_ratio,
        "angle_ratio": args.angle_ratio,
        "batch": args.batch_size,
        "iterations": args.iterations,
        "config": args.config,
        "lr": args.lr,
        "wd": args.wd,
        "swin": args.swin,
        "user": args.user,
        }
    
    wandb.init(
            # set the wandb project where this run will be logged
            project="tyrano",
            name = f'{args.name}_{args.current_time.strftime("%Y_%m_%d-%H_%M_%S")}_{args.user}_run{i}',
            # track hyperparameters and run metadata
            config= wandb_config
            )
