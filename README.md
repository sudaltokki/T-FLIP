# FLIP-KD
This repository contains the source code for the knowledge distillation of a fine-tuned CLIP model (FLIP) for the FAS (Face Anti-Spoofing) task.


## Dataset Preparation

### FLIP MCIO Dataset
We use the FLIP MCIO dataset for our experiments. Ensure the dataset is structured correctly and placed in the specified directories.

   ```
   data/MCIO/frame/
   |-- casia
       |-- train
       |   |--real
       |   |  |--1_1_frame0.png, 1_1_frame1.png 
       |   |--fake
       |      |--1_3_frame0.png, 1_3_frame1.png 
       |-- test
           |--real
           |  |--1_1_frame0.png, 1_1_frame1.png 
           |--fake
              |--1_3_frame0.png, 1_3_frame1.png 
   |-- msu
       |-- train
       |   |--real
       |   |  |--real_client002_android_SD_scene01_frame0.png, real_client002_android_SD_scene01_frame1.png
       |   |--fake
       |      |--attack_client002_android_SD_ipad_video_scene01_frame0.png, attack_client002_android_SD_ipad_video_scene01_frame1.png
       |-- test
           |--real
           |  |--real_client001_android_SD_scene01_frame0.png, real_client001_android_SD_scene01_frame1.png
           |--fake
              |--attack_client001_android_SD_ipad_video_scene01_frame0.png, attack_client001_android_SD_ipad_video_scene01_frame1.png
   |-- replay
       |-- train
       |   |--real
       |   |  |--real_client001_session01_webcam_authenticate_adverse_1_frame0.png, real_client001_session01_webcam_authenticate_adverse_1_frame1.png
       |   |--fake
       |      |--fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame1.png
       |-- test
           |--real
           |  |--real_client009_session01_webcam_authenticate_adverse_1_frame0.png, real_client009_session01_webcam_authenticate_adverse_1_frame1.png
           |--fake
              |--fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame1.png
   |-- oulu
       |-- train
       |   |--real
       |   |  |--1_1_01_1_frame0.png, 1_1_01_1_frame1.png
       |   |--fake
       |      |--1_1_01_2_frame0.png, 1_1_01_2_frame1.png
       |-- test
           |--real
           |  |--1_1_36_1_frame0.png, 1_1_36_1_frame1.png
           |--fake
              |--1_1_36_2_frame0.png, 1_1_36_2_frame1.png
   |-- celeb
       |-- real
       |   |--167_live_096546.jpg
       |-- fake
           |--197_spoof_420156.jpg       
   ```

## Inference with FLIP Teacher Model

To perform inference with the FLIP teacher model, use the following command:

```bash
python infer.py \
    --report_logger_path path/to/save/performance.csv \
    --config M \
    --method flip_mcl \
    --ckpt /home/jiwon/FLIP_yy/student/t_ckpt/oulu_flip_mcl.pth.tar
```

## Knowledge Distillation (Train Student Model)

To perform knowledge distillation, run the `experiment.sh` script with the following parameters:
```bash
sh experiment.sh
```

You can modify the parameter values in `experiment.sh`

```bash
--t-model-checkpoint "/home/jiwon/FLIP-KD/student/t_ckpt/msu_flip_mcl.pth.tar" \
--config custom \
--op_dir ckpt \
--report_logger_path logs \
--root "/home/jiwon/Dataset/FLIP_Dataset/MCIO/frame/" \
--dataroot "/home/jiwon/Dataset/FLIP_Dataset/MCIO/txt/" \
--epochs 100 \
--batch_size 3 \
--t_batch_size 30 \
--lr=0.000001 \
--alpha_ckd_loss 0. \
--alpha_fd_loss 0. \
--alpha_affinity_loss 0. \
--alpha_icl_loss 0.
```

## Configuration Details

- **Teacher Model**: ViT-B-16
- **Student Model**: ViT-T-16
- **Dataset**: FLIP MCIO
- **Preprocessing**: Follow the same preprocessing steps as used for FLIP.
