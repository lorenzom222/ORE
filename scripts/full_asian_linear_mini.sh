#!/usr/bin/env bash

# IMAGES_DIR=/home/nina/asian_skew_dataset
# IMAGES_DIR=/home/nina/swav/mini_dataset
# IMAGES_DIR=/data/datasets/Places_LT/small_easyformat/train
# IMAGES_DIR=/data/datasets/ImageNet-100/train
IMAGES_DIR=/home/lorenzo/ore-dir/swav/data/experiments/drive/final_data/home/nina/eval_dataset/

EXPT_NAME=EVAL_ASIAN_SKEW_FINETUNE
EXPT_PATH="./experiments/$EXPT_NAME"
mkdir -p $EXPT_PATH
GPU=2
CKPT="/home/lorenzo/ore-dir/swav/scripts/experiments/MobNet_FINETUNE_ASIAN_MINI/checkpoint.pth.tar"

#CHANGE SUBCLASS NUM and EXPT_NAMA and EXPT_PATH!!!!!
# Added --full_folder
CUDA_VISIBLE_DEVICES=${GPU} python ../eval_linear_mobilenet_INat.py \
    --epochs 1 \
    --ckpt_path $CKPT\
    --num_classes 40 \
    --pretrain_classes 40 \
    --data_path $IMAGES_DIR \
    --wandb False \
    --batch_size 64 \
    --track_race_acc True

