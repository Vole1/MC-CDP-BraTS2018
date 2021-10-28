import os
import random
import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import RMSprop

from datasets.brats_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef
from metrics_do import calculate_accuracy_and_confidence, brier_score, entropy, compute_mce_and_ece
from models.model_factory import make_model
from params import args

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    t0 = timeit.default_timer()

    predictions_repetition = args.times_sample_per_test
    cal_error_bins = args.cal_error_bins
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
                       pretrained_weights=args.pretrained_weights,
                       do_rate=args.dropout_rate)
    print("Building model {} from weights {} ".format(args.network, weights))
    print(f'Using dropout rate {args.dropout_rate}')
    model.load_weights(weights)

    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_generator((args.resize_size, args.resize_size), args.preprocessing_function,
                                            batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')

    print(f'Evaluating {weights} model')
    loss = make_loss(args.loss_function)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

    metrics = {args.loss_function: [],
               'binary_crossentropy': [],
               'hard_dice_coef_ch1': [],
               'hard_dice_coef': [],
               'brier_score': [],
               'expected_calibration_error': []
               }
    exclude_metrics_to_print_on_the_go = ['tf_brier_score', 'expected_calibration_error', 'maximum_calibration_error']

    loop_stop = data_generator.__len__()
    counter = -1
    prog_bar = tf.keras.utils.Progbar(loop_stop)
    for x, y in data_generator:
        counter += 1
        if counter >= loop_stop:
            break
        x_repeated = np.repeat(x, predictions_repetition, axis=0)
        predicts_x_repeated = model.predict(x_repeated, verbose=0)
        predicts_x = np.asarray([predicts_x_repeated[j*predictions_repetition:(j+1)*predictions_repetition, ...]
                                 for j in range(x.shape[0])])
        mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=1)

        batch_entropy_of_mean = entropy(mean_predicts)
        batch_mean_entropy = tf.reduce_mean(entropy(predicts_x), axis=1)
        mutual_info = batch_entropy_of_mean + batch_mean_entropy

        metrics[args.loss_function].append(loss(y, mean_predicts).numpy())
        metrics['binary_crossentropy'].append(binary_crossentropy(y, mean_predicts).numpy())
        metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, mean_predicts).numpy())
        metrics['hard_dice_coef'].append(hard_dice_coef(y, mean_predicts).numpy())
        metrics['brier_score'].append(brier_score(y, mean_predicts).numpy())
        metrics['expected_calibration_error'].append(calculate_accuracy_and_confidence(y.astype(np.int32),
                                                                                       mean_predicts,
                                                                                       mutual_info))

        prog_bar.update(counter+1, [(k, round(v[-1], 4)) for k,v in metrics.items() if k not in
                                    exclude_metrics_to_print_on_the_go])

    loss_value, bce_value, hdc1_value, hdc_value, brier_score_value = \
        Mean()(metrics[args.loss_function]), \
        Mean()(metrics['binary_crossentropy']), \
        Mean()(metrics['hard_dice_coef_ch1']), \
        Mean()(metrics['hard_dice_coef']), \
        Mean()(metrics['brier_score'])

    accs, confds = zip(*metrics['expected_calibration_error'])
    accs, confds = np.concatenate(np.asarray(accs), axis=0), \
                   np.concatenate(np.asarray(confds), axis=0)

    mce_value, correct_ece_value = compute_mce_and_ece(accs, confds, cal_error_bins)

    print(f'Performed {predictions_repetition} repetitions per sample')
    print(f'Dropout rate: {args.dropout_rate}')
    print(f'{weights} evaluation results:')
    print(f'{args.loss_function}: {loss_value:.4f}, '
          f'binary_crossentropy: {bce_value:.4f}, '
          f'hard_dice_coef_ch1: {hdc1_value:.4f}, '
          f'hard_dice_coef: {hdc_value:.4f}')
    print('Monte-Calro estimation')
    print(f'brier_score: {brier_score_value:.4f}, '
          f'\nexp_calibration_error: {correct_ece_value:.4f}',
          f'\nmax_calibration_error: {mce_value:.4f}')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    exit(0)


if __name__ == '__main__':
    main()
