import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def brier_score(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    return K.mean(K.pow(y_pred_f - y_true_f, 2))


def actual_accuracy_and_confidence(y_true, y_pred, uncertainty):
    acc = K.cast(y_true[..., 0] == K.round(y_pred[..., 0]), dtype='float32')
    #return acc, conf
    return acc, 1 - uncertainty[..., 0], y_pred[..., 0], y_true[..., 0]


def entropy(y_pred):
    return -y_pred * tf.math.log(y_pred + 1e-10)


def compute_TP_and_TN(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    TPs = (y_true[..., 0] == 1) & (y_pred_rounded[..., 0] == 1)
    TNs = (y_true[..., 0] == 0) & (y_pred_rounded[..., 0] == 0)
    return TPs, TNs


def compute_FTP_and_FTN(y_true, y_pred, uncertainty, thrds=(1, 0.8, 0.75, 0.6, 0.5, 0.4, 0.25, 0.2, 0.0)):
    y_pred_rounded = np.round(y_pred)
    TPs = (y_true[..., 0] == 1) & (y_pred_rounded[..., 0] == 1)
    TNs = (y_true[..., 0] == 0) & (y_pred_rounded[..., 0] == 0)
    if 1 not in thrds:
        thrds.append(1)
    FTPs = {}
    FTNs = {}
    for thrd in sorted(thrds):
        if thrd == 1:
            FTPs[thrd] = np.sum(TPs)
            FTNs[thrd] = np.sum(TNs)
        else:
            FTPs[thrd] = np.sum(TPs & (uncertainty[..., 0] < thrd).numpy())
            FTNs[thrd] = np.sum(TNs & (uncertainty[..., 0] < thrd).numpy())
    return FTPs, FTNs


def compute_filtered_hard_dice(y_true, y_pred, uncertainty, thrds=(1, 0.8, 0.75, 0.6, 0.5, 0.4, 0.25, 0.2, 0.0), smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    uncertainty_f = K.flatten(uncertainty[..., 0])
    filtered_hard_dices = {}
    for thrd in sorted(thrds):
        if thrd == 1:
            filtered_y_true_f = y_true_f
            filtered_y_pred_f = y_pred_f
        else:
            filtered_y_true_f = tf.where(uncertainty_f < thrd, y_true_f, 0)
            filtered_y_pred_f = tf.where(uncertainty_f < thrd, y_pred_f, 0)
        intersection = K.sum(filtered_y_true_f * filtered_y_pred_f)
        filtered_hard_dices[thrd] = 100. * (2. * intersection + smooth) / (K.sum(filtered_y_true_f) + K.sum(filtered_y_pred_f) + smooth)
    return filtered_hard_dices


#def compute_correct_ece(accs, confds, n_bins, pred_probs):
def compute_mce_and_correct_ece(accs, confds, n_bins, pred_probs, y_true):
    plot_x_pred_prob = []
    plot_x_conf = []
    plot_y = []
    bin_eces = []
    bin_mces = []
    accs = accs.flatten()
    confds = confds.flatten()
    pred_probs = pred_probs.flatten()
    y_true = y_true.flatten()
    probs_min = np.min(confds)
    h_w_wise_bins_len = (np.max(confds) - probs_min) / n_bins
    for j in range(n_bins):
        # tf.print(tf.convert_to_tensor(accs).shape, tf.convert_to_tensor(probs).shape)
        #print(f'\n---BORDERS of {j} bin:', probs_min + (h_w_wise_bins_len * j), probs_min + (h_w_wise_bins_len * (j + 1)))
        if j == 0:
            include_flags = np.logical_and(confds >= probs_min + (h_w_wise_bins_len * j), confds <= probs_min + (h_w_wise_bins_len * (j + 1)))
        else:
            include_flags = np.logical_and(confds > probs_min + (h_w_wise_bins_len * j), confds <= probs_min + (h_w_wise_bins_len * (j + 1)))
        if np.sum(include_flags) == 0:
            continue
        included_accs = accs[include_flags]
        included_probs = confds[include_flags]
        #print(np.unique(included_accs, return_counts=True))
        #print(np.unique(np.round(np.asarray(pred_probs[include_flags])) == np.asarray(y_true[include_flags]), return_counts=True))
        #print(np.unique(np.abs(np.asarray(pred_probs[include_flags]) - np.asarray(y_true[include_flags]))<=0.25, return_counts=True))
        a = (np.abs(np.asarray(pred_probs[include_flags]) - np.asarray(y_true[include_flags])))
        #print(np.min(a), np.max(a))
        mean_accuracy = included_accs.mean()
        #print(tf.reduce_mean(included_accs))
        mean_confidence = included_probs.mean()
        bin_scaled_ece = np.abs(mean_accuracy-mean_confidence)*np.sum(include_flags, axis=-1)
        bin_eces.append(bin_scaled_ece)
        bin_mces.append(np.abs(mean_accuracy-mean_confidence))

        plot_x_pred_prob.append(pred_probs[include_flags].mean())
        plot_x_conf.append(mean_confidence)
        plot_y.append(mean_accuracy)
    pixel_wise_ece = np.sum(np.asarray(bin_eces), axis=0) / accs.shape[-1]
    #print('\nPixel-wise eces:\n', np.asarray(bin_eces)/accs.shape[-1])
    #print('\nX pred_prob:\n', np.asarray(plot_x_pred_prob))
    #print('\nX conf:\n', np.asarray(plot_x_conf))
    #print('\nY:\n', np.asarray(plot_y))
    return max(bin_mces), pixel_wise_ece.mean()



def computeMI(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi
