import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--channels', type=int, default="3")
arg('--gpu', default=None)
arg('--epochs', type=int, default=100)
arg('--freeze_till_layer', default='input_1')
arg('--preprocessing_function')
arg('--weights')
arg('--ensemble_weights_pattern')
arg('--pretrained_weights', default='imagenet')
arg('--learning_rate', type=float, default=0.001)
arg('--dropout_rate', type=float, default=0.3)
arg('--crop_size', type=int, default=None) #192)
arg('--resize_size', type=int, default=None)
arg('--crops_per_image', type=int, default=1)
arg('--batch_size', type=int, default=16)
arg('--num_workers', type=int, default=7)
arg('--loss_function', default='bce_dice')
arg('--optimizer', default="rmsprop")
arg('--clr')
arg('--decay', type=float, default=0.0)
arg('--save_period', type=int, default=1)
arg('--network', default='densenet_unet')
arg('--alias', default='')
arg('--steps_per_epoch', type=int, default=0)
arg('--times_sample_per_test', type=int, default=20)
arg('--use_softmax', action="store_true")
arg('--use_full_masks', action="store_true")
arg('--ensemble_type', default='None')
arg('--models_count', type=int, default=5)
# arg('--multi_gpu', action="store_true")
arg('--seed', type=int, default=777)
arg('--tf_seed', type=int, default=None)
arg('--models_dir', default='nn_models')
arg('--images_dir', default='../data/image')
arg('--masks_dir', default='../data/mask')
arg('--log_dir', default='xception_fpn')
arg('--test_images_dir', default='../data_test/images')
arg('--test_masks_dir', default='../data_test/masks')
arg('--out_root_dir', default='./../predictions')
arg('--out_masks_folder')
arg('--models',  nargs='+')
arg('--out_channels',  type=int, default=2)

args = parser.parse_args()

if args.pretrained_weights == "None":
    args.pretrained_weights = None
