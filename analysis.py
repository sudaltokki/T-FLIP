import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import umap
import os
import json
import base64
from io import BytesIO
import PIL.Image as Image
import plotly.graph_objects as go
from alphashape import alphashape
from third_party.utils.random import random_seed
import os
from PIL import Image
import matplotlib.pyplot as plt

# 기타 필요한 모듈 임포트
from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset, get_dataset_one_to_one_ssl_clip, get_dataset_ssl_clip
from student.fas import flip_mcl, flip_v, flip_it
from teacher.config import (
    configC, configM, configI, configO, config_cefa, config_surf, config_wmca,
    config_CI, config_CO, config_CM, config_MC, config_MI, config_MO,
    config_IC, config_IO, config_IM, config_OC, config_OI, config_OM
)
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from student.params import parse_args

from third_party.utils.random import random_seed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


def generate_random_color():
    return f'rgba({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)}, 0.3)'

def create_image_grid(image_paths, grid_size=(2, 5), image_size=(128, 128)):
    """
    이미지 경로 리스트를 받아서 지정된 크기와 그리드로 이미지를 배치합니다.
    :param image_paths: 이미지 경로 리스트
    :param grid_size: 그리드의 (행, 열) 크기
    :param image_size: 각 이미지의 (width, height) 크기
    :return: 배치된 이미지를 반환
    """
    grid_width, grid_height = grid_size
    img_width, img_height = image_size

    # 새 이미지 생성
    grid_img = Image.new('RGB', (grid_width * img_width, grid_height * img_height))

    for i, image_path in enumerate(image_paths):
        if i >= grid_width * grid_height:
            break  # 그리드에 더 이상 이미지를 넣을 수 없을 때

        img = Image.open(image_path)
        img = img.resize(image_size)

        # 위치 계산
        x = (i % grid_width) * img_width
        y = (i // grid_width) * img_height

        # 이미지 붙이기
        grid_img.paste(img, (x, y))

    return grid_img

def show_images(correct_filenames, incorrect_filenames, grid_size=(2, 5), image_size=(128, 128)):
    """
    Correct와 Incorrect 이미지들을 각각 그리드에 나열하여 보여줍니다.
    :param correct_filenames: Correct 이미지 파일 경로 리스트
    :param incorrect_filenames: Incorrect 이미지 파일 경로 리스트
    :param grid_size: 그리드의 (행, 열) 크기
    :param image_size: 각 이미지의 (width, height) 크기
    """
    correct_grid = create_image_grid(correct_filenames, grid_size, image_size)
    incorrect_grid = create_image_grid(incorrect_filenames, grid_size, image_size)

    # 두 개의 이미지를 matplotlib로 표시
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(correct_grid)
    axs[0].set_title('Correct Predictions')
    axs[0].axis('off')

    axs[1].imshow(incorrect_grid)
    axs[1].set_title('Incorrect Predictions')
    axs[1].axis('off')

    plt.show()


def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

def plot_with_hovertemplate(image_features_np, filenames, correct_filenames, incorrect_filenames):
    # t-SNE 적용하여 2D로 변환
    tsne = TSNE(n_components=2,perplexity=15, random_state=42)
    tsne_results = tsne.fit_transform(image_features_np)

    # 올바르게 예측된 파일과 잘못 예측된 파일 분류
    correct_indices = [i for i, filename in enumerate(filenames) if filename in correct_filenames]
    incorrect_indices = [i for i, filename in enumerate(filenames) if filename in incorrect_filenames]

    # Plotly에서 시각화
    fig = go.Figure()

    # 올바르게 예측된 데이터 플롯
    for i in correct_indices:
        fig.add_trace(go.Scatter(
            x=[tsne_results[i, 0]], y=[tsne_results[i, 1]],
            mode='markers',
            marker=dict(size=10, color='green'),
            hovertemplate=f'<b>Filename:</b> {filenames[i]}<br>',
        ))

    # 잘못 예측된 데이터 플롯
    for i in incorrect_indices:
        fig.add_trace(go.Scatter(
            x=[tsne_results[i, 0]], y=[tsne_results[i, 1]],
            mode='markers',
            marker=dict(size=10, color='red'),
            hovertemplate=f'<b>Filename:</b> {filenames[i]}<br>',
        ))

    
    fig.update_layout(
        title="Feature Space with t-SNE and Hover Images",
        xaxis_title="t-SNE Feature 1",
        yaxis_title="t-SNE Feature 2",
        showlegend=False
    )

    fig.show()

def plot_with_hovertemplate_umap2d(image_features_np, filenames, correct_filenames, incorrect_filenames, keywords):
    # UMAP 적용하여 2D로 변환
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(image_features_np)

    fig = go.Figure()

    # 올바르게 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=umap_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 0],
        y=umap_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 1],
        mode='markers',
        marker=dict(size=5, color='pink'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in correct_filenames]
    ))

    # 잘못 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=umap_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 0],
        y=umap_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 1],
        mode='markers',
        marker=dict(size=5, color='purple'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in incorrect_filenames]
    ))

    for keyword, fillcolor in keywords.items():
        print(keyword)
        indices = [i for i, filename in enumerate(filenames) if keyword in filename]
        if len(indices) > 2:  # 최소 3개의 점이 있어야 다각형을 그릴 수 있음
            points = umap_results[indices, :]

            hull = ConvexHull(points)
            hull_points = np.append(hull.vertices, hull.vertices[0])  # 점들을 닫아서 다각형을 만듦

            fig.add_trace(go.Scatter(
                x=points[hull_points, 0],
                y=points[hull_points, 1],
                mode='lines',
                fill='toself',
                opacity=0.2,
                line=dict(color='rgba(0,0,0,0)'),  # 라인을 보이지 않게 설정
                fillcolor=fillcolor,
                hoverinfo='skip',
                name=f'Group: {keyword}',
                visible='legendonly'
            ))

    fig.update_layout(
        title="Feature Space with UMAP and 2D Hover Images",
        xaxis_title="UMAP Feature 1",
        yaxis_title="UMAP Feature 2",
        showlegend=True
    )

    fig.show()

def plot_with_hovertemplate_umap2d2(image_features_np, filenames, correct_filenames, incorrect_filenames, real_list, spoof_list, keywords):
    # UMAP 적용하여 2D로 변환
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(image_features_np)

    fig = go.Figure()

    # 올바르게 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=umap_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 0],
        y=umap_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 1],
        mode='markers',
        marker=dict(size=5, color='red'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in correct_filenames]
    ))

    # 잘못 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=umap_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 0],
        y=umap_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 1],
        mode='markers',
        marker=dict(size=5, color='blue'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in incorrect_filenames]
    ))

    # 올바르게 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=umap_results[[i for i in range(len(filenames)) if filenames[i] in real_list], 0],
        y=umap_results[[i for i in range(len(filenames)) if filenames[i] in real_list], 1],
        mode='markers',
        marker=dict(size=5, color='pink'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in real_list]
    ))

    # 잘못 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=umap_results[[i for i in range(len(filenames)) if filenames[i] in spoof_list], 0],
        y=umap_results[[i for i in range(len(filenames)) if filenames[i] in spoof_list], 1],
        mode='markers',
        marker=dict(size=5, color='purple'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in spoof_list]
    ))

    fig.update_layout(
        title="Feature Space with UMAP and 2D Hover Images",
        xaxis_title="UMAP Feature 1",
        yaxis_title="UMAP Feature 2",
        showlegend=True
    )

    fig.show()

def plot_with_hovertemplate_3d(image_features_np, filenames, correct_filenames, incorrect_filenames,keywords):
    # t-SNE 적용하여 3D로 변환
    tsne = TSNE(n_components=3, perplexity=50, random_state=42)
    tsne_results = tsne.fit_transform(image_features_np)

    fig = go.Figure()

    # 올바르게 예측된 데이터 플롯
    fig.add_trace(go.Scatter3d(
        x=tsne_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 0],
        y=tsne_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 1],
        z=tsne_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 2],
        mode='markers',
        marker=dict(size=2, color='green'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in correct_filenames]
    ))

    # 잘못 예측된 데이터 플롯
    fig.add_trace(go.Scatter3d(
        x=tsne_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 0],
        y=tsne_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 1],
        z=tsne_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 2],
        mode='markers',
        marker=dict(size=2, color='red'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in incorrect_filenames]
    ))

    for keyword, color in keywords.items():
        print(keyword)
        indices = [i for i, filename in enumerate(filenames) if keyword in filename]
        if len(indices) > 2:  # 최소 3개의 점이 있어야 다각형을 그릴 수 있음
            points = tsne_results[indices, :]

            hull = ConvexHull(points)
            vertices = hull.simplices
            x, y, z = points.T

            fig.add_trace(go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=vertices[:, 0],
                j=vertices[:, 1],
                k=vertices[:, 2],
                opacity=0.2,
                color=color,
                hoverinfo='skip',
                name=f'Group: {keyword}',
                visible='legendonly',
            ))

    fig.update_layout(
        title="Feature Space with t-SNE and 3D Hover Images",
        scene=dict(
            xaxis_title="t-SNE Feature 1",
            yaxis_title="t-SNE Feature 2",
            zaxis_title="t-SNE Feature 3"
        ),
        showlegend=True
    )

    fig.show()

def plot_with_hovertemplate_2d(image_features_np, filenames, correct_filenames, incorrect_filenames,keywords):
    # t-SNE 적용하여 2D로 변환
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(image_features_np)

    fig = go.Figure()

    # 올바르게 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=tsne_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 0],
        y=tsne_results[[i for i in range(len(filenames)) if filenames[i] in correct_filenames], 1],
        mode='markers',
        marker=dict(size=5, color='green'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in correct_filenames]
    ))

    # 잘못 예측된 데이터 플롯
    fig.add_trace(go.Scatter(
        x=tsne_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 0],
        y=tsne_results[[i for i in range(len(filenames)) if filenames[i] in incorrect_filenames], 1],
        mode='markers',
        marker=dict(size=5, color='red'),
        hovertemplate='<b>Filename:</b> %{text}<br>',
        text=[filenames[i] for i in range(len(filenames)) if filenames[i] in incorrect_filenames]
    ))

    for keyword, fillcolor in keywords.items():
        print(keyword)
        indices = [i for i, filename in enumerate(filenames) if keyword in filename]
        if len(indices) > 2:  # 최소 3개의 점이 있어야 다각형을 그릴 수 있음
            points = tsne_results[indices, :]

            hull = ConvexHull(points)
            hull_points = np.append(hull.vertices, hull.vertices[0])  # 점들을 닫아서 다각형을 만듦

            fig.add_trace(go.Scatter(
                x=points[hull_points, 0],
                y=points[hull_points, 1],
                mode='lines',
                fill='toself',
                opacity=0.2,
                line=dict(color='rgba(0,0,0,0)'),  # 라인을 보이지 않게 설정
                fillcolor=fillcolor,
                hoverinfo='skip',
                name=f'Group: {keyword}',
                visible='legendonly'
            ))

    fig.update_layout(
        title="Feature Space with t-SNE and 2D Hover Images",
        xaxis_title="t-SNE Feature 1",
        yaxis_title="t-SNE Feature 2",
        showlegend=True
    )

    fig.show()


def eval_and_analyze(args):
    # 모델 및 데이터셋 로드
    random_seed()
    with open(os.path.join(os.getcwd(), 'student/model_config/'+args.t_model+'.json'), 'r') as f:
        args.t_embed_dim = json.load(f)['embed_dim']
    with open(os.path.join(os.getcwd(), 'student/model_config/'+args.model+'.json'), 'r') as f:
        args.s_embed_dim = json.load(f)['embed_dim']

    config_map = {
        'I': configI, 'C': configC, 'M': configM, 'O': configO,
        'cefa': config_cefa, 'surf': config_surf, 'wmca': config_wmca
    }
    config = config_map[args.config]
    config.checkpoint = args.ckpt

    valid_dataloader = get_dataset_ssl_clip(
        args, config.src1_data, config.src1_train_num_frames, config.src2_data,
        config.src2_train_num_frames, config.src3_data, config.src3_train_num_frames,
        config.src4_data, config.src4_train_num_frames, config.src5_data,
        config.src5_train_num_frames, config.tgt_data, tgt_test_num_frames=config.tgt_test_num_frames,
        collate_fn=custom_collate_fn
    )[-1]

    model = flip_mcl(args, device, in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device)
    if args.model != 'ViT-B-16':
        ckpt = torch.load(config.checkpoint)
        model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # 평가 및 분석
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    output_dict_tmp = {}
    target_dict_tmp = {}
    correct_filenames = []
    incorrect_filenames = []
    filenames = []
    image_features_list = []
    text_features_list = []
    dataset_labels_list = []
    class_labels_list = []

    with torch.no_grad():
        for iter, (input, target, videoID, name) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            cls_out = model.forward_eval(input, True)

            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()

            for i in range(len(prob)):
                filenames.append(name[i])
                if videoID[i] in prob_dict:
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = [prob[i]]
                    label_dict[videoID[i]] = [label[i]]
                    output_dict_tmp[videoID[i]] = [cls_out[i].view(1, 2)]
                    target_dict_tmp[videoID[i]] = [target[i].view(1)]

            # forward_tsne로 image 및 text features 추출
            image_features, text_features = model.forward_tsne(input, True)

            image_features_list.append(image_features.cpu().numpy())
            text_features_list.append(text_features.cpu().numpy())
            dataset_labels_list.append(np.full(image_features.size(0), i))
            class_labels_list.append(target.cpu().numpy())

    print('infer finished!')
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)

        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        loss = criterion(avg_single_video_output, avg_single_video_target.long())
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])

    image_features_np = np.concatenate(image_features_list, axis=0)
    text_features_np = np.concatenate(text_features_list, axis=0)
    dataset_labels_np = np.concatenate(dataset_labels_list, axis=0)
    class_labels_np = np.concatenate(class_labels_list, axis=0)

    for i in range(len(prob_list)):
        print('problist',prob_list[i])
        if abs(prob_list[i] - label_list[i]) > 0.6:
            incorrect_filenames.append(filenames[i])
        else:
            correct_filenames.append(filenames[i])
    
    real_list = []
    spoof_list = []

    for i in range(len(label_list)):
        if label_list[i] == 0:
            spoof_list.append(filenames[i])
        else:
            real_list.append(filenames[i])


    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    print('threshold', threshold)
    print('ACC', ACC_threshold)
    fpr, tpr, thr = roc_curve(label_list, prob_list)
    print('fpr', fpr)
    print('tpr', tpr)
    print('thr', thr)


    tpr_filtered = tpr[fpr <= 1 / 100]
    rate = tpr_filtered[-1] if len(tpr_filtered) > 0 else 0
    print("TPR@FPR = ", rate)

    # 파일명 저장
    with open(f'{args.config}_incorrect_filenames.txt', 'w') as f:
        for filename in incorrect_filenames:
            f.write(f"{filename}\n")

    with open(f'{args.config}_correct_filenames.txt', 'w') as f:
        for filename in correct_filenames:
            f.write(f"{filename}\n")

    keywords = {'real': 'pink', 'fake': 'blue','printed': 'yellow','ipad': 'orange','iphone': 'green',}  # pink for 'real', blue for 'fake'
    #keywords = {'real': 'pink', 'fake': 'blue','HR': 'yellow'}
    plot_with_hovertemplate_umap2d2(image_features_np, filenames, correct_filenames, incorrect_filenames, spoof_list, real_list,keywords)
    #plot_with_hovertemplate_2d(image_features_np, filenames, correct_filenames, incorrect_filenames,keywords)
    #show_images(correct_filenames, incorrect_filenames)
    #plot_with_hovertemplate_3d(image_features_np, filenames, correct_filenames, incorrect_filenames,keywords)

    # 결과 리턴
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate
    ], [prob_list, label_list]


def main(args):
    random_seed()
    args = parse_args(args)
    results, prob_label_lists = eval_and_analyze(args)

if __name__ == '__main__':
    main(sys.argv[1:])

    
