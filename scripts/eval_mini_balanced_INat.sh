#!/usr/bin/env bash

# IMAGES_DIR=/home/nina/asian_skew_dataset
# IMAGES_DIR=/home/nina/swav/mini_dataset
# IMAGES_DIR=/data/datasets/Places_LT/small_easyformat/train
# IMAGES_DIR=/data/datasets/ImageNet-100/train
IMAGES_DIR=/home/lorenzo/ore-dir/INat_data/mini_data/linear_prob_dataset
EXPT_NAME=INat_balanced_flip_crop_sl
EXPT_PATH="./experiments/INat_balanced_flip_crop_sl"
mkdir -p $EXPT_PATH
GPU=2

#CHANGE SUBCLASS NUM and EXPT_NAMA and EXPT_PATH!!!!!
# Added --full_folder
CUDA_VISIBLE_DEVICES=${GPU} python eval_linear_mobilenet_INat.py \
    --epochs 1 \
    --ckpt_path /home/lorenzo/ore-dir/swav/experiments/INat_balanced_flip_crop_sl/checkpoint.pth.tar \
    --num_classes 6 \
    --pretrain_classes 270 \
    --full_folder $IMAGES_DIR \
    --wandb False \
    --batch_size 64 \
    --track_supcat_acc True
