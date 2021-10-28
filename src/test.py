import os
import random
import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

from datasets.brats_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef
from models.model_factory import make_model
from params import args

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    t0 = timeit.default_timer()

    weights = os.path.join(args.models_dir, args.model)
    if args.crop_size:
        print('Using crops of shape ({}, {})'.format(args.crop_size, args.crop_size))
        input_shape = (args.crop_size, args.crop_size, args.channels)
    elif args.resize_size:
        print('Using resizes of shape ({}, {})'.format(args.resize_size, args.resize_size))
        input_shape = (args.resize_size, args.resize_size, args.channels)
    else:
        print('Using sized images (256, 256)')
        input_shape = (256, 256, args.channels)

    model = make_model(args.network,
                       input_shape,
                       pretrained_weights=args.pretrained_weights)
    print("Building model {} from weights {} ".format(args.network, weights))
    model.load_weights(weights)

    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')

    print(f'Evaluating {weights} model')

    model.compile(loss=make_loss(args.loss_function),
              optimizer=optimizer,
              metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

    test_loss = model.evaluate(data_generator, verbose=1)
    print(f'{weights} evaluation results:')
    for el in list(zip([args.loss_function, 'binary_crossentropy', 'hard_dice_coef_ch1', 'hard_dice_coef'], test_loss)):
        print(f'{el[0]}: {el[1]}')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))


if __name__ == '__main__':
    main()
