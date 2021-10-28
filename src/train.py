import os
from datetime import datetime

import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from datasets.brats_binary import DSB2018BinaryDataset
from losses import make_loss, hard_dice_coef, hard_dice_coef_ch1
from models.model_factory import make_model
from params import args

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


if args.tf_seed is not None:
    tf.random.set_seed(args.tf_seed)

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index + 1]:
            l.trainable = False


def main():
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
                       pretrained_weights=args.pretrained_weights,
                       do_rate=args.dropout_rate,
                       total_training_steps=args.epochs*args.steps_per_epoch)
    if args.weights is None:
        print('No full model weights passed')
    else:
        weights_path = os.path.join(args.models_dir, args.weights)
        print('Loading weights from {}'.format(weights_path))
        model.load_weights(weights_path, by_name=True)
    freeze_model(model, args.freeze_till_layer)
    optimizer = RMSprop(lr=args.learning_rate)
    if args.optimizer:
        if args.optimizer == 'rmsprop':
            optimizer = RMSprop(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'adam':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'amsgrad':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay), amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True, decay=float(args.decay))
    dataset = DSB2018BinaryDataset(args.images_dir, args.masks_dir, args.channels, seed=args.seed)
    train_generator = dataset.train_generator((args.crop_size, args.crop_size),
                                              (args.resize_size, args.resize_size),
                                              args.preprocessing_function,
                                              batch_size=args.batch_size)
    val_generator = dataset.val_generator((args.resize_size, args.resize_size),
                                          args.preprocessing_function,
                                          batch_size=args.batch_size)
    best_model_file = '{}/best_{}_{}.h5'.format(args.models_dir, args.alias, args.network)

    best_model = ModelCheckpoint(filepath=best_model_file, monitor='val_loss',
                                 verbose=1,
                                 mode='min',
                                 save_freq='epoch',
                                 save_best_only=True,
                                 save_weights_only=True)
    last_model_file = '{}/last_{}_{}.h5'.format(args.models_dir, args.alias, args.network)

    last_model = ModelCheckpoint(filepath=last_model_file, monitor='val_loss',
                                 verbose=1,
                                 mode='min',
                                 save_freq=int(args.save_period)*args.steps_per_epoch,
                                 save_best_only=False,
                                 save_weights_only=True)
    model.compile(loss=make_loss(args.loss_function),
                  optimizer=optimizer,
                  metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

    # callbacks = [best_model, last_model]
    callbacks = [best_model]

    tb_log_dir_path = "logs/{}_{}".format(args.log_dir, datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    tb = TensorBoard(tb_log_dir_path)
    print(f"Saving tb logs to {tb_log_dir_path}")
    callbacks.append(tb)
    steps_per_epoch = len(dataset.train_ids) / args.batch_size + 1
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    validation_data = val_generator
    validation_steps = len(dataset.val_ids) //val_generator.batch_size

    model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            max_queue_size=5,
            verbose=1,
            workers=args.num_workers)


if __name__ == '__main__':
    main()
