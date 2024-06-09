torchrun --nproc_per_node 1 -m \
    --master_addr=127.0.0.2 --master_port=29140 \
    student.main_kd \
    --t_model ViT-B-16 \
    --model ViT-T-16 \
    --t-model-checkpoint "/home/jiwon/FLIP-KD/student/t_ckpt/msu_flip_mcl.pth.tar"\
    --config custom \
    --op_dir ckpt \
    --report_logger_path logs \
    --root "/home/jiwon/Dataset/FLIP_Dataset/MCIO/frame/"\
    --dataroot "/home/jiwon/Dataset/FLIP_Dataset/MCIO/txt/"\
    --epochs 100 \
    --batch_size 3 \
    --t_batch_size 30 \
    --lr=0.000001 \
    --alpha_ckd_loss 0. \
    --alpha_fd_loss 0. \
    --alpha_affinity_loss 0. \
    --alpha_icl_loss 0.
