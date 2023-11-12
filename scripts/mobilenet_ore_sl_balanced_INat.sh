#!/usr/bin/env bash
IMAGES_DIR=/home/lorenzo/ore-dir/INat_data/mini_data/balanced_data
# IMAGES_DIR=/home/nina/asian_skew_dataset
# IMAGES_DIR=/home/nina/swav/mini_dataset
# IMAGES_DIR=/data/datasets/Places_LT/small_easyformat/train
# IMAGES_DIR=/data/datasets/ImageNet-100/train
EXPT_NAME=INat_balanced_flip_crop_sl
EXPT_PATH="./experiments/INat_balanced_flip_crop_sl"
mkdir -p $EXPT_PATH
GPU=1

#CHANGE SUBCLASS NUM and EXPT_NAMA and EXPT_PATH!!!!!

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=1 main_mobilenet_sl_INat.py \
  --data_path ${IMAGES_DIR} \
  --arch mobilenet_v3_large \
  --epochs 1 \
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
  --max_class 1083 \
  --nmb_prototypes 1000 \
  --workers 8 \
  --checkpoint_freq 1 \
  --dump_path $EXPT_PATH \
  --run_name $EXPT_NAME \
  --swav_aug false \
  --subset_size 270 \
  --wandb false \
  --image_net_normalize true \
  --lr_scheduler true \

#balanced classes = 1113
#skewed insect dataset = 2249
#skewed plant dataset = 2249 
#skewed birds dataset = 2249 
