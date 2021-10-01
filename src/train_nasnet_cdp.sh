#!/usr/bin/env bash

##################### Nasnet Concrete Droppath ##############################

python train.py \
--channels 4 \
--network nasnet_cdp \
--pretrained_weights imagenet \
--num_workers 8  \
--alias nii_mc_cdp \
--freeze_till_layer input_1  \
--loss double_head_loss \
--optimizer adam  \
--learning_rate 0.0001  \
--decay 0.0001  \
--batch_size 10 \
--steps_per_epoch 500 \
--epochs 100 \
--preprocessing_function caffe \
--images_dir ../data/images \
--masks_dir ../data/masks \
--log_dir nasnet_mc_cdp_nii
