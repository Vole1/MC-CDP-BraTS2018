import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def brier_score(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    return K.mean(K.pow(y_pred_f - y_true_f, 2))


def calculate_accuracy_and_confidence(y_true, y_pred, uncertainty):
    acc = K.cast(y_true[..., 0] == K.round(y_pred[..., 0]), dtype='float32')
    return acc, 1 - uncertainty[..., 0]


def entropy(y_pred):
    return -y_pred * tf.math.log(y_pred + 1e-10)


def compute_mce_and_ece(accs, confds, n_bins):
    bin_eces = []
    bin_mces = []
    accs = accs.flatten()
    confds = confds.flatten()
    probs_min = np.min(confds)
    h_w_wise_bins_len = (np.max(confds) - probs_min) / n_bins
    for j in range(n_bins):
        if j == 0:
            include_flags = np.logical_and(confds >= probs_min + (h_w_wise_bins_len * j), confds <= probs_min + (h_w_wise_bins_len * (j + 1)))
        else:
            include_flags = np.logical_and(confds > probs_min + (h_w_wise_bins_len * j), confds <= probs_min + (h_w_wise_bins_len * (j + 1)))
        if np.sum(include_flags) == 0:
            continue
        included_accs = accs[include_flags]
        included_probs = confds[include_flags]
        mean_accuracy = included_accs.mean()
        mean_confidence = included_probs.mean()
        bin_scaled_ece = np.abs(mean_accuracy-mean_confidence)*np.sum(include_flags, axis=-1)
        bin_eces.append(bin_scaled_ece)
        bin_mces.append(np.abs(mean_accuracy-mean_confidence))
    pixel_wise_ece = np.sum(np.asarray(bin_eces), axis=0) / accs.shape[-1]
    return max(bin_mces), pixel_wise_ece.mean()

