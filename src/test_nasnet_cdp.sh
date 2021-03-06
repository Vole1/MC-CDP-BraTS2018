#!/usr/bin/env bash

##################### Nasnet Concrete Droppath ##############################

python test_with_do.py \
--channels 4 \
--num_workers 8 \
--network nasnet_cdp \
--pretrained_weights None \
--preprocessing_function caffe \
--dropout_rate -1  \
--batch_size 1 \
--times_sample_per_test 20 \
--out_channels 2 \
--test_images_dir ../data_test/images \
--test_masks_dir ../data_test/masks \
--model best_mc_nasnet_cdp.h5

