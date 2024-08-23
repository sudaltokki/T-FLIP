import sys

# sys.path.append('../../')

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset
from utils.dataset import get_dataset_one_to_one_ssl_clip , get_dataset_for_tsne
from student.fas import flip_mcl, flip_v, flip_it
import random
import numpy as np
from teacher.config import configTSNE
from datetime import datetime
import time
from timeit import default_timer as timer
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
from torch.autograd import Variable


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import importlib_metadata
import umap

def plot_tsne(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):
    """
    features: (N, D) shape의 feature embeddings
    dataset_labels: 각 embedding에 대한 데이터셋 레이블 (N,)
    class_labels: 각 embedding에 대한 클래스 레이블 (N,)
    datasets: 사용된 데이터셋의 이름 리스트
    num_classes: 클래스의 수 (보통 Real/Fake 2가지)
    """

    # t-SNE 적용
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 시각화
    plt.figure(figsize=(12, 10))
    colors = plt.cm.get_cmap("tab10", len(datasets) * num_classes)

    for i, dataset in enumerate(datasets):
        for j in range(num_classes):
            idx = (dataset_labels == i) & (class_labels == j)
            print(i, j)
            label_name = f"{dataset} - {'real' if j == 1 else 'spoof'}"
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color=colors(i * num_classes + j), label=label_name, alpha=0.6)
    
    plt.legend()
    plt.suptitle("t-SNE visualization of embeddings by Dataset and Class")
    plt.title(file_name)

    plt.savefig(f'tsne_result/tsne_{type}_features_visualization_{file_name}.png')
    plt.show()

def plot_tsne_3d(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):
    """
    features: (N, D) shape의 feature embeddings
    dataset_labels: 각 embedding에 대한 데이터셋 레이블 (N,)
    class_labels: 각 embedding에 대한 클래스 레이블 (N,)
    datasets: 사용된 데이터셋의 이름 리스트
    num_classes: 클래스의 수 (보통 Real/Fake 2가지)
    """

    # t-SNE 적용
    tsne = TSNE(n_components=3, perplexity=70, random_state=42)
    features_3d = tsne.fit_transform(features)

    # 3D 시각화
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap("tab10", len(datasets) * num_classes)

    for i, dataset in enumerate(datasets):
        for j in range(num_classes):
            idx = (dataset_labels == i) & (class_labels == j)
            print(i, j)
            label_name = f"{dataset} - {'real' if j == 1 else 'spoof'}"
            ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2], 
                       color=colors(i * num_classes + j), label=label_name, alpha=0.6)

    ax.legend()

    ax.set_title("t-SNE 3D visualization of embeddings by Dataset and Class")
    plt.title(file_name)

    plt.savefig(f'tsne_result/tsne_{type}_features_visualization_3d_{file_name}.png')
    plt.show()

def plot_umap(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):
    # UMAP 적용
    reducer = umap.UMAP(n_components=2, random_state=42)
    features_2d = reducer.fit_transform(features)

    # 시각화
    plt.figure(figsize=(12, 10))
    colors = plt.cm.get_cmap("tab10", len(datasets) * num_classes)

    for i, dataset in enumerate(datasets):
        for j in range(num_classes):
            idx = (dataset_labels == i) & (class_labels == j)
            print(i, j)
            label_name = f"{dataset} - {'real' if j == 1 else 'spoof'}"
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color=colors(i * num_classes + j), label=label_name, alpha=0.6)
    
    plt.legend()

    plt.title(file_name)
    plt.suptitle("UMAP visualization of embeddings by Dataset and Class")

    plt.savefig(f'tsne_result/umap_{type}_features_visualization_{file_name}.png')
    plt.show()

def infer(config, args):

    args_info = "\n".join([f"{arg}: {getattr(args, arg)}" for arg in vars(args)])
    logging.info(f"\n-----------------------------Arguments----------------------------|\n{args_info}\n")
  
    # 데이터셋 로드
    src1_train_dataloader, src2_train_dataloader, src3_train_dataloader, src4_train_dataloader = get_dataset_for_tsne(args,  
        config.src1_data, config.src1_train_num_frames, config.src2_data,
        config.src2_train_num_frames, config.src3_data,
        config.src3_train_num_frames, config.src4_data,
        config.src4_train_num_frames, collate_fn=custom_collate_fn)
    data_loaders_list = get_dataset_for_tsne(args,  
        config.src1_data, config.src1_train_num_frames, config.src2_data,
        config.src2_train_num_frames, config.src3_data,
        config.src3_train_num_frames, config.src4_data,
        config.src4_train_num_frames, collate_fn=custom_collate_fn)
    #data_loaders_list = data_loaders_list[:-1]
    #data_loaders_list = data_loaders_list[0]
    random_seed(42, 0)

    model = flip_mcl(args, device, in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device) # ssl applied to image, and euclidean distance applied to image and text cosine similarity

    ckpt = torch.load(config.checkpoint)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print('load checkpoint')

    
    # feature 추출 및 t-SNE 시각화
    image_features_list = []
    text_features_list = []
    dataset_labels_list = []
    class_labels_list = []
    
    dataset_names = ['oulu', 'casia', 'msu', 'replay']  # 실제 데이터셋 이름을 반영하세요

    with torch.no_grad():
        for i, data_loader in enumerate(data_loaders_list):
            print(i)
            for iter, (input, target, videoID, name) in enumerate(data_loader):
                input = input.cuda()
                target = torch.from_numpy(np.array(target)).long().cuda()

                # forward_tsne로 image 및 text features 추출
                image_features, text_features = model.forward_tsne(input, True)

                image_features_list.append(image_features.cpu().numpy())
                text_features_list.append(text_features.cpu().numpy())
                dataset_labels_list.append(np.full(image_features.size(0), i))
                class_labels_list.append(target.cpu().numpy())

    # list를 numpy 배열로 변환
    image_features_np = np.concatenate(image_features_list, axis=0)
    text_features_np = np.concatenate(text_features_list, axis=0)
    dataset_labels_np = np.concatenate(dataset_labels_list, axis=0)
    class_labels_np = np.concatenate(class_labels_list, axis=0)

    split_ckpt = args.ckpt.split('/')
    file_name = split_ckpt[1]+'_'+split_ckpt[3].split('_')[3]+split_ckpt[3].split('_')[4]
    
    plot_tsne(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name)
    plot_tsne_3d(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name)
    #plot_umap(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name)
    #plot_tsne(text_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='text', file_name=file_name)
        
    return 


if __name__ == '__main__':
  args = sys.argv[1:]

  random_seed()

  args = parse_args(args)

  with open(os.path.join(os.getcwd(), 'student/model_config/'+args.t_model+'.json'), 'r') as f:
        args.t_embed_dim = json.load(f)['embed_dim']
  with open(os.path.join(os.getcwd(), 'student/model_config/'+args.model+'.json'), 'r') as f:
      args.s_embed_dim = json.load(f)['embed_dim']


  if args.config == 'TSNE':
      config = configTSNE


  for attr in dir(config):
    if attr.find('__') == -1:
      print('%s = %r' % (attr, getattr(config, attr)))

  config.checkpoint = args.ckpt

  infer(config, args)
  
