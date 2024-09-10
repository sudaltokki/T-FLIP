import sys

# sys.path.append('../../')

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset
from utils.dataset import get_dataset_one_to_one_ssl_clip , get_dataset_for_tsne
from train.fas import flip_mcl
import random
import numpy as np
from teacher.config import configTSNE
import time
from timeit import default_timer as timer
import os
import torch
import torch.nn as nn
import argparse
from train.params import parse_args
import clip
import logging
from utils.logger import setup_logging
from train.fas import flip_mcl
import json
import wandb
from torch.autograd import Variable
from third_party.utils.random import random_seed


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


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
from sklearn.preprocessing import StandardScaler




def plot_tsne(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):

    perplexity = 50
    # t-SNE 적용
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features)
    print(f"features shape: {features.shape}")
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

    plt.savefig(f'result/tsne_{type}_features_visualization_{file_name}_p{perplexity}.png')
    plt.show()
    
def plot_tsne_combined(image_features, text_features, dataset_labels, class_labels, datasets, file_name, num_classes=2, perplexity=30):

    # t-SNE 적용
    scaler = StandardScaler()
    image_features = scaler.fit_transform(image_features)
    text_features = scaler.fit_transform(text_features)


    # 결합된 피처에 대해 t-SNE 적용 (2차원)
    tsne = TSNE(n_components=2,perplexity=perplexity, random_state=42)
    image_2d = tsne.fit_transform(image_features)

    # 시각화
    plt.figure(figsize=(12, 10))
    colors = plt.cm.get_cmap("tab10", len(datasets) * num_classes)

    for i, dataset in enumerate(datasets):
        for j in range(num_classes):
            idx = (dataset_labels == i) & (class_labels == j)
            label_name = f"{dataset} - {'real' if j == 1 else 'spoof'} (Images)"
            plt.scatter(image_2d[idx, 0], image_2d[idx, 1], color=colors(i * num_classes + j), label=label_name, alpha=0.6)

    classes = ['spoof', 'real']
    colors = ['pink', 'yellow']

    text_2d = tsne.fit_transform(text_features)
    
    for i in range(num_classes):
        plt.scatter(text_2d[i, 0], text_2d[i, 1], color=colors[i], label=classes[i], alpha=0.6)

    plt.title(file_name)
    plt.legend()
    plt.suptitle("t-SNE visualization of embeddings by Dataset and Class")
    plt.savefig(f'result/tsne_image&text_features_visualization_{file_name}_p{perplexity}.png')
    plt.show()


def plot_tsne_3d(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2, perplexity=30):
    """
    features: (N, D) shape의 feature embeddings
    dataset_labels: 각 embedding에 대한 데이터셋 레이블 (N,)
    class_labels: 각 embedding에 대한 클래스 레이블 (N,)
    datasets: 사용된 데이터셋의 이름 리스트
    num_classes: 클래스의 수 (보통 Real/Fake 2가지)
    """

    # t-SNE 적용
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
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

    plt.savefig(f'result/tsne_{type}_features_visualization_3d_{file_name}_p{perplexity}.png')
    plt.show()

def plot_tsne_3d_combined(image_features, text_features, dataset_labels, class_labels, datasets, file_name, num_classes=2, perplexity=30):
    """
    features: (N, D) shape의 feature embeddings
    dataset_labels: 각 embedding에 대한 데이터셋 레이블 (N,)
    class_labels: 각 embedding에 대한 클래스 레이블 (N,)
    datasets: 사용된 데이터셋의 이름 리스트
    num_classes: 클래스의 수 (보통 Real/Fake 2가지)
    """

    # t-SNE 적용
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    features_3d = tsne.fit_transform(image_features)

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

    classes = ['spoof', 'real']
    colors = ['pink', 'yellow']

    text_3d = tsne.fit_transform(text_features)
    
    for i in range(num_classes):
        ax.scatter(text_3d[i, 0], text_3d[i, 1], text_3d[i, 2], color=colors[i], label=classes[i], alpha=0.6)

    ax.legend()

    ax.set_title("t-SNE 3D visualization of embeddings by Dataset and Class")
    plt.title(file_name)

    plt.savefig(f'result/tsne3d_{type}_features_visualization_3d_{file_name}_p{perplexity}.png')
    plt.show()    

def plot_umap_3d(image_features, dataset_labels, class_labels, datasets, file_name, num_classes=2, min_dist=0.8):
    
    # UMAP for image features
    reducer_img = umap.UMAP(n_components=3, random_state=42, min_dist=min_dist)
    image_3d = reducer_img.fit_transform(image_features)

    # Create dataframes for both image and text features
    df_img = pd.DataFrame(image_3d, columns=['x', 'y', 'z'])
    df_img['dataset'] = [datasets[int(label)] for label in dataset_labels]
    df_img['class'] = ['real' if int(label) == 1 else 'spoof' for label in class_labels]
    
    
    # Plot the combined UMAP
    fig = px.scatter_3d(df_img, x='x', y='y', z='z', color='dataset', symbol='class', 
                        title=f"Combined UMAP 3D visualization of Image and Text Features_mindist{min_dist}")

    fig.update_traces(marker=dict(size=1))
    fig.show()    

def plot_umap(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2, min_dist=0.8):
    # UMAP 적용
    reducer = umap.UMAP(min_dist=min_dist, n_components=2, random_state=42)
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

    plt.savefig(f'result/umap_{type}_features_visualization_{file_name}_mindist{min_dist}.png')
    plt.show()

def plotly_umap_2d(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):
    reducer = umap.UMAP(spread=1.5, n_components=2, random_state=42)
    features_2d = reducer.fit_transform(features)

    df = pd.DataFrame(features_2d, columns=['x', 'y'])
    df['dataset'] = [datasets[int(label)] for label in dataset_labels]
    df['class'] = ['real' if int(label) == 1 else 'spoof' for label in class_labels]

    fig = px.scatter(df, x='x', y='y', color='dataset', symbol='class', 
                     title="UMAP 2D visualization by Dataset and Class")
    fig.update_traces(marker=dict(size=5))
    fig.show()

import plotly.express as px
import umap
import pandas as pd

# UMAP 적용
def plotly_umap(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):
    reducer = umap.UMAP(n_components=3, random_state=42)
    features_3d = reducer.fit_transform(features)

    df = pd.DataFrame(features_3d, columns=['x', 'y', 'z'])
    df['dataset'] = [datasets[int(label)] for label in dataset_labels]
    df['class'] = ['real' if int(label) == 1 else 'spoof' for label in class_labels]

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='dataset', symbol='class', 
                        title="UMAP 3D visualization by Dataset and Class")
    fig.update_traces(marker=dict(size=1))
    fig.show()

# UMAP 적용
def plotly_tsne(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):
    perplexity = 40
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    features_3d = tsne.fit_transform(features)

    df = pd.DataFrame(features_3d, columns=['x', 'y', 'z'])
    df['dataset'] = [datasets[int(label)] for label in dataset_labels]
    df['class'] = ['real' if int(label) == 1 else 'spoof' for label in class_labels]

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='dataset', symbol='class', 
                        title="UMAP 3D visualization by Dataset and Class")
    fig.update_traces(marker=dict(size=1))
    fig.show()


import holoviews as hv
from holoviews import opts
import umap

hv.extension('bokeh')
def holoviews_umap(features, dataset_labels, class_labels, datasets, type, file_name, num_classes=2):
    reducer = umap.UMAP(n_components=3, random_state=42)
    features_3d = reducer.fit_transform(features)

    # 데이터셋 이름과 클래스 라벨을 추가한 데이터 생성
    data = np.column_stack((features_3d, dataset_labels, class_labels))

    # Points3D로 3D 시각화 수행
    points = hv.Scatter3D((data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]), vdims=['dataset', 'class'])
    points.opts(color='dataset', cmap='Category10', size=5, tools=['hover'], width=600, height=600)

    hv.show(points)

def infer(config, args):

    args_info = "\n".join([f"{arg}: {getattr(args, arg)}" for arg in vars(args)])
    logging.info(f"\n-----------------------------Arguments----------------------------|\n{args_info}\n")
  
    # 데이터셋 로드
    data_loaders_list = get_dataset_for_tsne(args,  
        config.src1_data, config.src1_train_num_frames, config.src2_data,
        config.src2_train_num_frames, config.src3_data,
        config.src3_train_num_frames, config.src4_data,
        config.src4_train_num_frames, collate_fn=custom_collate_fn)
    #data_loaders_list = data_loaders_list[:-1]
    #data_loaders_list = data_loaders_list[0]
    

    
    model = flip_mcl(args, device, in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device) # ssl applied to image, and euclidean distance applied to image and text cosine similarity
    
    if args.model != 'ViT-B-16':
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
                text_features = text_features.cpu().numpy()
                dataset_labels_list.append(np.full(image_features.size(0), i))
                class_labels_list.append(target.cpu().numpy())
    
    # list를 numpy 배열로 변환
    image_features_np = np.concatenate(image_features_list, axis=0)
    dataset_labels_np = np.concatenate(dataset_labels_list, axis=0)
    class_labels_np = np.concatenate(class_labels_list, axis=0)
    

    if args.model != 'ViT-B-16':
        split_ckpt = args.ckpt.split('/')
        file_name = split_ckpt[1]+'_'+split_ckpt[3].split('_')[3]+split_ckpt[3].split('_')[4]
    else:
        split_ckpt = args.t_model_checkpoint.split('/')
        file_name = split_ckpt[4].split('.')[0]

    #plot_tsne(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name)
    #plot_tsne_3d(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name)
    #plotly_umap(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name)
    #plot_umap(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name,min_dist=0.3)
    #plot_umap(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name, min_dist=0.5)

    plot_umap_3d(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, file_name=file_name)

    #holoviews_umap(image_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='image', file_name=file_name)
    #plot_tsne(text_features_np, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, type='text', file_name=file_name)
    #plot_tsne_combined(image_features_np, text_features, dataset_labels_np, class_labels_np, datasets=dataset_names, num_classes=2, file_name=file_name, perplexity=10)



    return 

def main(args):
    random_seed()

    args = parse_args(args)

    with open(os.path.join(os.getcwd(), 'train/model_config/'+args.t_model+'.json'), 'r') as f:
          args.t_embed_dim = json.load(f)['embed_dim']
    with open(os.path.join(os.getcwd(), 'train/model_config/'+args.model+'.json'), 'r') as f:
        args.s_embed_dim = json.load(f)['embed_dim']


    if args.config == 'TSNE':
        config = configTSNE


    for attr in dir(config):
      if attr.find('__') == -1:
        print('%s = %r' % (attr, getattr(config, attr)))

    config.checkpoint = args.ckpt

    infer(config, args)

if __name__ == '__main__':
    main(sys.argv[1:])

  
  
