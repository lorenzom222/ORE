#!/usr/bin/env bash
IMAGES_DIR=/home/lorenzo/ore-dir/swav/data/experiments/drive/final_data/home/nina/eval_dataset/
EXPT_NAME=Full-SwAV-Balanced-Linear
EXPT_PATH="./experiments/Full/Full-SwAV-Balanced-Linear"
mkdir -p $EXPT_PATH
GPU=0,1,2,3

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=4 ../eval_linear_ssl.py \
  --dump_path $EXPT_PATH \
  --data_path $IMAGES_DIR \
  --arch mobilenet_v3_large \
  --epochs 100 \
  --batch_size 64 \
  --scheduler_type cosine \
  --pretrained /home/lorenzo/ore-dir/swav/scripts/experiments/Full/Full-SwAV-Balanced/checkpoints/SwAV_MobNet_Ckpt_1.pth \
