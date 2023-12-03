#!/usr/bin/env bash

# IMAGES_DIR=/home/nina/asian_skew_dataset
# IMAGES_DIR=/home/nina/swav/mini_dataset
# IMAGES_DIR=/data/datasets/Places_LT/small_easyformat/train
# IMAGES_DIR=/data/datasets/ImageNet-100/train
IMAGES_DIR=/home/lorenzo/ore-dir/swav/data/experiments/drive/final_data/home/nina/eval_dataset/

EXPT_NAME=EVAL_ResNet_ORE_ASIAN_512
EXPT_PATH="./experiments/$EXPT_NAME"
mkdir -p $EXPT_PATH
GPU=2
CKPT="/home/lorenzo/ore-dir/swav/resnet/ResNet_ORE_ASIAN_PRETRAIN_512/checkpoint.pth.tar"

#CHANGE SUBCLASS NUM and EXPT_NAMA and EXPT_PATH!!!!!
# Added --full_folder
CUDA_VISIBLE_DEVICES=${GPU} python ../eval_resnet.py \
    --epochs 100 \
    --ckpt_path $CKPT\
    --num_classes 40 \
    --pretrain_classes 5930 \
    --data_path $IMAGES_DIR \
    --wandb True \
    --batch_size 64 \
    --track_race_acc True\
    --run_name $EXPT_NAME\
    --save True


