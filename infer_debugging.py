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
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


def overlay(image, attention_map):

    attention_map = attention_map / attention_map.max()
    cmap = plt.get_cmap('jet')
    attention_map_colored = cmap(attention_map)[:, :, :3]

    image = transforms.ToTensor()(image).permute(1, 2, 0).numpy()
    attention_map_resized = np.array(Image.fromarray((attention_map_colored * 255).astype(np.uint8)).resize(image.shape[:2], Image.BILINEAR)) / 255

    overlay = (0.5 * image + 0.5 * attention_map_resized)
    overlay = np.clip(overlay, 0, 1)

    return overlay


def save_image(path, attention_maps, output_dir):

    try:
        img = Image.open(path).convert("RGB")
        image_filename = os.path.basename(path)
        image_base, _ = os.path.splitext(image_filename)

        image_output_dir = os.path.join(output_dir, image_base)
        os.makedirs(image_output_dir, exist_ok=True)

        img.save(os.path.join(image_output_dir, f"{image_base}_original.jpg"))

        for idx, attention_map in enumerate(attention_maps):

            overlay_img = overlay(img, attention_map)
            plt.imshow(overlay_img)
            overlay_dir = os.path.join(image_output_dir, f"layer_{idx}.jpg")
            plt.savefig(overlay_dir, bbox_inches='tight', pad_inches=0)
            plt.close()
        
    except Exception as e:
        print(f"failed to save attention maps for {path}: {e}")


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
    valid_args = eval(test_dataloader, net1, norm_flag=True, vis=args.vis)
    # judge model according to HTER
    is_best = valid_args[3] <= best_model_HTER
    best_model_HTER = min(valid_args[3], best_model_HTER)
    threshold = valid_args[5]

    best_model_ACC = valid_args[6]
    best_model_AUC = valid_args[4]
    best_TPR_FPR = valid_args[-3]
            

    return best_model_HTER*100.0, best_model_AUC*100.0, best_TPR_FPR*100.0, valid_args[8], valid_args[9]


def main(args):

    args = parse_args(args)

    with open(os.path.join(os.getcwd(), 'train/model_config/'+args.t_model+'.json'), 'r') as f:
        args.t_embed_dim = json.load(f)['embed_dim']
    with open(os.path.join(os.getcwd(), 'train/model_config/'+args.model+'.json'), 'r') as f:
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
        hter, auc, tpr_fpr, true_false_list, attention_maps = infer(args, config)

        f.write(f'{hter},{auc},{tpr_fpr}\n')

    output_directory = "debugging_images"
    output_directory = os.path.join(args.report_logger_path, args.name, output_directory)
    os.makedirs(output_directory, exist_ok=True)

    for i, image_path in enumerate(true_false_list):
        save_image(image_path, attention_maps[i], output_directory)


if __name__ == "__main__":
    main(sys.argv[1:])