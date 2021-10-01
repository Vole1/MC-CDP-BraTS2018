import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
#from tensorflow.python.keras.utils.tf_utils import smart_cond
from tensorflow.python.framework.smart_cond import smart_cond


class DropPath(Layer):
    """Applies Droppath to the input.
        The Dropout layer randomly sets input units to 0 with a frequency of `rate`
        at each step during training time, which helps prevent overfitting.
        Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
        all inputs is unchanged.
        Note that the Dropout layer only applies when `training` is set to True
        such that no values are dropped during inference. When using `model.fit`,
        `training` will be appropriately set to True automatically, and in other
        contexts, you can set the kwarg explicitly to True when calling the layer.
        (This is in contrast to setting `trainable=False` for a Dropout layer.
        `trainable` does not affect the layer's behavior, as Dropout does
        not have any variables/weights that can be frozen during training.)
        Arguments:
            paths_rate: Float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
                For instance, if your inputs have shape
                `(batch_size, timesteps, features)` and
                you want the dropout mask to be the same for all timesteps,
                you can use `noise_shape=(batch_size, 1, features)`.
            seed: A Python integer to use as random seed.
        Call arguments:
            inputs: Input tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).
    """ #TODO: refactor docs

    def __init__(self, paths_rate, drop_paths_mask, seed=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_rate = paths_rate
        self.drop_paths_mask = drop_paths_mask
        self.seed = seed

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            drop_paths_count = np.sum(self.drop_paths_mask)
            rand_tens = tf.compat.v1.distributions.Bernoulli(1 - self.drop_rate).sample(sample_shape=len(inputs))
            preserve_mask = 1 - tf.cast(self.drop_paths_mask, tf.int32)
            rand_tens = tf.clip_by_value(rand_tens + preserve_mask, clip_value_min=0, clip_value_max=1)
            if not tf.reduce_any(tf.gather(rand_tens, tf.where(self.drop_paths_mask)[:, 0]) == 1):
                index_to_preserve = tf.where(self.drop_paths_mask)[:, 0][tf.random.uniform(shape=(),
                                                                                           maxval=drop_paths_count,
                                                                                           dtype=tf.int32)]
                rand_tens += [int(i == index_to_preserve) for i in range(len(inputs))]
            else:
                index_to_preserve = None
            # tf.print(self.drop_paths_mask)
            # tf.print('rnd_tens:', rand_tens)
            scaled_rand_tens = tf.cast(rand_tens / tf.reduce_sum(rand_tens), dtype=tf.float32)
            # tf.print(scaled_rand_tens)
            outputs = [tf.multiply(a, b) for a, b in zip(inputs, tf.unstack(scaled_rand_tens))]
            return outputs

        output = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))

        # tf.print('Maxes before droppath', [tf.math.reduce_max(inp_path) for inp_path in inputs])
        # tf.print('Maxes after droppath', [tf.math.reduce_max(out_path) for out_path in output])
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'drop_paths_mask': self.drop_paths_mask,
            'seed': self.seed
        }
        base_config = super(DropPath, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
