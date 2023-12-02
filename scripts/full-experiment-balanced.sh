#!/usr/bin/env bash
IMAGES_DIR=/home/lorenzo/ore-dir/home/nina/ore_balanced_dataset
EXPT_NAME=Full-SwAV-Balanced
EXPT_PATH="./experiments/Full/Full-SwAV-Balanced"
mkdir -p $EXPT_PATH
GPU=0,1,2,3

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=4 ../main_mobilenet_ssl.py \
  --data_path ${IMAGES_DIR} \
  --arch mobilenet_v3_large \
  --epochs 2 \
  --wd 5e-4 \
  --warmup_epochs 8 \
  --start_warmup 0 \
  --batch_size 64 \
  --base_lr 2.08 \
  --final_lr 0.0006 \
  --size_crops 224 128 \
  --nmb_crops 2 6 \
  --min_scale_crops 0.3 0.05 \
  --max_scale_crops 1. 0.3 \
  --use_fp16 true \
  --epsilon 0.03 \
  --freeze_prototypes_niters 5005 \
  --queue_length 384 \
  --epoch_queue_starts 15 \
  --hidden_mlp 960 \
  --feat_dim 128 \
  --max_class 1960 \
  --nmb_prototypes 1000 \
  --workers 8 \
  --checkpoint_freq 1 \
  --dump_path $EXPT_PATH \
  --run_name $EXPT_NAME \
  --swav_aug false \
  --subset_size 1960 \
  --wandb true \
  --image_net_normalize true \
  --lr_scheduler true \

