import os
import pickle
import random
import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import RMSprop

from datasets.brats_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef
from metrics_do import actual_accuracy_and_confidence, brier_score, entropy, compute_mce_and_correct_ece, \
    compute_FTP_and_FTN, compute_filtered_hard_dice, compute_TP_and_TN
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
    weights = [os.path.join(args.models_dir, m) for m in args.models]
    if args.crop_size:
        print('Using crops of shape ({}, {})'.format(args.crop_size, args.crop_size))
        input_shape = (args.crop_size, args.crop_size, args.channels)
    elif args.resize_size:
        print('Using resizes of shape ({}, {})'.format(args.resize_size, args.resize_size))
        input_shape = (args.resize_size, args.resize_size, args.channels)
    else:
        print('Using sized images (256, 256)')
        input_shape = (256, 256, args.channels)

    models = []
    for w in weights:
        model = make_model(args.network,
                           input_shape,
                           pretrained_weights=args.pretrained_weights,
                           do_rate=args.dropout_rate)
        print("Building model {} from weights {} ".format(args.network, w))
        print(f'Using dropout rate {args.dropout_rate}')
        model.load_weights(w)
        models.append(model)

    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')

    for i, model in enumerate(models):
        print(f'Evaluating {weights[i]} model')
        loss = make_loss(args.loss_function)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

        metrics = {args.loss_function: [],
                   'binary_crossentropy': [],
                   'hard_dice_coef_ch1': [],
                   'hard_dice_coef': [],
                   'brier_score': [],
                   'expected_calibration_error': [],
                   'thresholded_hard_dice': [],
                   'FTP': [],
                   'FTN': [],
                   'TP, TN, unc': []
                   }
        exclude_metrics_to_print_on_the_go = ['tf_brier_score', 'expected_calibration_error',
                                              'maximum_calibration_error', 'thresholded_hard_dice', 'FTP', 'FTN',
                                              'TP, TN, unc']

        loop_stop = data_generator.__len__()

        counter = -1
        data_gen_len = data_generator.__len__()
        entropy_of_mean = []
        mean_entropy = []
        prog_bar = tf.keras.utils.Progbar(data_gen_len)
        thrds = sorted(np.round(np.arange(0, 1, 0.025), 4), reverse=True) + [0.75, 0.25]
        for x, y in data_generator:
            counter += 1
            if counter >= loop_stop:
                break
            x_repeated = np.repeat(x, predictions_repetition, axis=0)
            predicts_x_repeated = model.predict(x_repeated, verbose=0)
            predicts_x = np.asarray([predicts_x_repeated[j*predictions_repetition:(j+1)*predictions_repetition, ...] for j in range(x.shape[0])])
            mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=1)

            batch_mean_entropy = tf.reduce_mean(entropy(predicts_x), axis=1)
            batch_entropy_of_mean = entropy(mean_predicts)
            mutual_info = tf.abs(batch_mean_entropy - batch_entropy_of_mean)

            metrics[args.loss_function].append(loss(y, mean_predicts).numpy())
            metrics['binary_crossentropy'].append(binary_crossentropy(y, mean_predicts).numpy())
            metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, mean_predicts).numpy())
            metrics['hard_dice_coef'].append(hard_dice_coef(y, mean_predicts).numpy())
            metrics['brier_score'].append(brier_score(y, mean_predicts).numpy())

            metrics['expected_calibration_error'].append(actual_accuracy_and_confidence(y.astype(np.int32), mean_predicts, mutual_info))
            metrics['thresholded_hard_dice'].append(compute_filtered_hard_dice(y, mean_predicts, mutual_info, thrds=thrds))
            FTPs, FTNs = compute_FTP_and_FTN(y, mean_predicts, mutual_info, thrds=thrds)
            metrics['FTP'].append(FTPs)
            metrics['FTN'].append(FTNs)
            metrics['TP, TN, unc'].append([*compute_TP_and_TN(y, mean_predicts), mutual_info[..., 0]])

            mean_entropy.append(tf.reduce_mean(entropy(predicts_x[..., 0]), axis=1))
            entropy_of_mean.append(entropy(mean_predicts[..., 0]))

            prog_bar.update(counter+1, [(k, round(v[-1], 4)) for k,v in metrics.items() if k not in exclude_metrics_to_print_on_the_go])

        loss_value, bce_value, hdc1_value, hdc_value, brier_score_value = \
            Mean()(metrics[args.loss_function]), \
            Mean()(metrics['binary_crossentropy']), \
            Mean()(metrics['hard_dice_coef_ch1']), \
            Mean()(metrics['hard_dice_coef']), \
            Mean()(metrics['brier_score'])

        ece_bins = 20

        accs, confds, pred_probs, y_true = zip(*metrics['expected_calibration_error'])
        accs, confds, pred_probs, y_true = np.concatenate(np.asarray(accs), axis=0), \
                                           np.concatenate(np.asarray(confds), axis=0), \
                                           np.concatenate(np.asarray(pred_probs), axis=0), \
                                           np.concatenate(np.asarray(y_true), axis=0)
        mce_value, correct_ece_value = compute_mce_and_correct_ece(accs, confds, ece_bins, pred_probs, y_true)

        F_dice = {k: np.mean([metrics['thresholded_hard_dice'][j][k] for j in range(len(metrics['thresholded_hard_dice']))]) for k in metrics['thresholded_hard_dice'][0].keys()}

        FTPs = {k: np.sum([metrics['FTP'][j][k] for j in range(len(metrics['FTP']))]) for k in metrics['FTP'][0].keys()}
        ratio_of_FTPs = {k: (FTPs[1] - FTPs[k]) / FTPs[1] if FTPs[1] > 0 else 0 for k in FTPs.keys()}
        FTNs = {k: np.sum([metrics['FTN'][j][k] for j in range(len(metrics['FTN']))]) for k in metrics['FTN'][0].keys()}
        ratio_of_FTNs = {k: (FTNs[1] - FTNs[k]) / FTNs[1] if FTNs[1] > 0 else 0 for k in FTNs.keys()}

        tp_tn_unc = np.asarray(metrics['TP, TN, unc'])
        TPs = {}
        TNs = {}
        for thrd in sorted(thrds):
            TPs[thrd] = np.sum(np.where(tp_tn_unc[:, 2] < thrd, tp_tn_unc[:, 0], 0))
            TNs[thrd] = np.sum(np.where(tp_tn_unc[:, 2] < thrd, tp_tn_unc[:, 1], 0))

        mean_entropy_subtr = np.mean(np.asarray(mean_entropy)-np.asarray(entropy_of_mean))

        print(f'Performed {predictions_repetition} repetitions per sample')
        print(f'Dropout rate: {args.dropout_rate}')
        print(f'{weights[i]} evaluation results:')
        print(f'{args.loss_function}: {loss_value:.4f}, '
              f'binary_crossentropy: {bce_value:.4f}, '
              f'hard_dice_coef_ch1: {hdc1_value:.4f}, '
              f'hard_dice_coef: {hdc_value:.4f}')
        print('Monte-Calro estimation')
        print(f'brier_score: {brier_score_value:.4f}, '
              f'\nexp_calibration_error: {correct_ece_value:.4f}',
              f'\nmax_calibration_error: {mce_value:.4f}',
              f'\nDices: '+'\t'.join([f'{k}: {v:.4f}' for k, v in F_dice.items()]),
              f'\nratios of FTPs: '+'\t'.join([f'{k}: {v:.4f}' for k, v in ratio_of_FTPs.items()]),
              f'\nratios of FTNs: '+'\t'.join([f'{k}: {v:.4f}' for k, v in ratio_of_FTNs.items()]),
              f'\nmean_entropy_subtr: {mean_entropy_subtr:.4f}')

        with open(os.path.join('..', 'predictions', 'data_to_visualize.pkl'), 'wb') as f:
            pickle.dump((F_dice, ratio_of_FTPs, ratio_of_FTNs), f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    exit(0)


def load_model_weights(w):
    model = make_model(args.network, (None, None, args.channels), pretrained_weights=args.pretrained_weights)
    donor_model = make_model(args.network[:-3], (None, None, args.channels), pretrained_weights=args.pretrained_weights)
    print("Building dropout model {} from weights without dropout {} ".format(args.network, w))
    donor_model.load_weights(w)

    j = 0
    for i, l in enumerate(model.layers):
        if j >= len(donor_model.layers):
            break
        d_l = donor_model.layers[j]
        if 'dropout' in l.name and 'dropout' not in d_l.name:
            continue

        j += 1
        for (w, d_w) in zip(l.weights, d_l.weights):
            w.assign(d_w)

    assert j == len(donor_model.layers)
    return model


if __name__ == '__main__':
    main()
