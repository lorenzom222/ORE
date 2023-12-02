#!/usr/bin/env bash
# IMAGES_DIR=/home/nina/ore_balanced_dataset
# IMAGES_DIR="/home/lorenzo/ore-dir/swav/data/experiments/full/unbalanced_asian"
IMAGES_DIR="/home/lorenzo/ore-dir/swav/data/experiments/drive/final_data/home/nina/ore_balanced_dataset"

# IMAGES_DIR=/home/nina/swav/mini_dataset
# IMAGES_DIR=/data/datasets/Places_LT/small_easyformat/train
# IMAGES_DIR=/data/datasets/ImageNet-100/train
EXPT_NAME=MobNet_FINETUNE_ASIAN_MINI
EXPT_PATH="./experiments/$EXPT_NAME"
mkdir -p $EXPT_PATH
GPU=0,1,2,3

#CHANGE SUBCLASS NUM and EXPT_NAMA and EXPT_PATH!!!!!

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=4 ../main_mobilenet_sl.py \
  --pretrained true \
  --data_path ${IMAGES_DIR} \
  --arch mobilenet_v3_large \
  --epochs 100 \
  --wd 5e-4 \
  --warmup_epochs 0 \
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
  --max_class 5940 \
  --nmb_prototypes 1000 \
  --workers 8 \
  --checkpoint_freq 20 \
  --dump_path $EXPT_PATH \
  --run_name $EXPT_NAME \
  --swav_aug false \
  --subset_size 40 \
  --wandb true \
  --image_net_normalize true \
  --lr_scheduler true \

#balanced classes = 1960
#skewed indian dataset = 5940
#skewed asian dataset = 5930 cleaned some anime data "m.03j2_6g m.0771g6 m.047qyhl m.09fhg4 m.05b4d7v m.0bpq7j m.03gw28c m.0dbp2s m.04jc6bw m.07f4lv"
#skewed african dataset = 5938 cleaned anime data "m.0809ry4 m.02rmg13"
#skewed caucasian dataset = 5935 cleaned anime data "m.03j2zfm m.03j2_6g m.09fhg4 m.0771g6 m.02rmg13"