import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.framework.smart_cond import smart_cond


class ScheduledDropout(Layer):
    """Applies Scheduled Dropout to the input.
        The Dropout layer randomly sets input units to 0 with a frequency of `rate`
        scheduled by network layer's depth and training step at each step, which
        helps prevent overfitting.
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
            drop_rate: Float between 0 and 1. Fraction of the input units to drop.
            cell_num: Cell number in the network
            total_num_cells: Number of cells in the network
            total_training_steps: Number of total steps performed during training
            seed: A Python integer to use as random seed.
        Call arguments:
            inputs: Input tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).
    """
    #TODO: debug the layer code
    def __init__(self, drop_rate, cell_num, total_num_cells, total_training_steps, seed=None, **kwargs):
        super(ScheduledDropout, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        self._cell_num = cell_num
        self._total_num_cells = total_num_cells
        self._total_training_steps = total_training_steps
        self.seed = seed

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        scheduled_drop_rate = self._compute_scheduled_dropout_rate()

        def dropped_inputs():
            noise_shape = [tf.shape(input=inputs)[0], 1, 1, 1]
            random_tensor = 1 - scheduled_drop_rate
            random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
            binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)
            keep_prob_inv = tf.cast(1.0 / (1-scheduled_drop_rate), inputs.dtype)
            outputs = inputs * keep_prob_inv * binary_tensor
            tf.Assert(tf.convert_to_tensor(
                tf.reduce_sum(binary_tensor)) >=
                      tf.convert_to_tensor(tf.reduce_sum(
                          tf.cast(tf.reduce_max(outputs, axis=[1,2,3]) > 0, dtype=tf.float32))),
                      data=['Nothing'])  # TODO: remove when debugged
            return outputs

        output = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))

        return output

    def _compute_scheduled_dropout_rate(self):
        drop_rate = self.drop_rate
        if self._total_num_cells is not None:  #drop_connect_version in ['v2', 'v3']:
            # Scale keep prob by layer number
            assert self._cell_num != -1
            # The added 2 is for the reduction cells
            num_cells = self._total_num_cells
            layer_ratio = (self._cell_num + 1) / float(num_cells)
            drop_rate = layer_ratio * drop_rate
        if self._total_training_steps is not None:  #drop_connect_version in ['v1', 'v3']:
            # Decrease the keep probability over time
            current_step = tf.convert_to_tensor(tf.compat.v1.train.get_or_create_global_step())
            tf.compat.v1.get_variable_scope().reuse_variables()
            current_step = tf.cast(current_step, tf.float32)
            drop_path_burn_in_steps = self._total_training_steps
            current_ratio = current_step / drop_path_burn_in_steps
            current_ratio = tf.minimum(1.0, current_ratio)
            drop_rate = current_ratio * drop_rate
        return drop_rate

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'cell_num': self._cell_num,
            'total_num_cells': self._total_num_cells,
            'total_training_steps': self._total_training_steps,
            'seed': self.seed
        }
        base_config = super(ScheduledDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ScheduledDroppath(Layer):
    """Applies Scheduled Droppath to the input.
        The Droppath layer randomly sets whole input path inside to 0 with a
        frequency of `rate` scheduled by network layer's depth and training
        step at each step, which helps prevent overfitting.
        Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
        all inputs is unchanged.
        Note that the Scheduled Droppath layer only applies when `training` is set to True.
        When using `model.fit`, `training` will be appropriately set to True
        automatically, and in other contexts, you can set the kwarg explicitly
        to True when calling the layer. (This is in contrast to setting
        `trainable=False` for a Droppath layer. `trainable` does not affect the
        layer's behavior, as Droppath does not have any variables/weights that
        can be frozen during training.)
        Arguments:
            drop_rate: Float between 0 and 1. Fraction of the inputs to drop.
            cell_num: Cell number in the network
            total_num_cells: Number of cells in the network
            total_training_steps: Number of total steps performed during training
            seed: A Python integer to use as random seed.
        Call arguments:
            inputs: Input tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, drop_rate, cell_num, total_num_cells, total_training_steps, seed=None, **kwargs):
        super(ScheduledDroppath, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        self._cell_num = cell_num
        self._total_num_cells = total_num_cells
        self._total_training_steps = total_training_steps
        self.seed = seed

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        scheduled_drop_rate = self._compute_scheduled_drop_rate()

        def dropped_inputs():
            noise_shape = [tf.shape(input=inputs)[0], 1, 1, 1]
            random_tensor = 1 - scheduled_drop_rate
            random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
            binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)
            keep_prob_inv = tf.cast(1.0 / (1-scheduled_drop_rate), inputs.dtype)
            tf.print('binary tesnor:', binary_tensor)
            outputs = inputs * keep_prob_inv * binary_tensor
            return outputs

        if training:
            output = dropped_inputs()
        else:
            output = tf.identity(inputs)
        return output

    def _compute_scheduled_drop_rate(self):
        drop_rate = self.drop_rate
        if self._total_num_cells is not None:
            # Scale keep prob by layer number
            assert self._cell_num != -1
            # The added 2 is for the reduction cells
            num_cells = self._total_num_cells
            layer_ratio = (self._cell_num + 1) / float(num_cells)
            drop_rate = layer_ratio * drop_rate
        if self._total_training_steps is not None:
            # Decrease the keep probability over time
            current_step = tf.convert_to_tensor(tf.compat.v1.train.get_or_create_global_step())
            tf.compat.v1.get_variable_scope().reuse_variables()
            current_step = tf.cast(current_step, tf.float32)
            drop_path_burn_in_steps = self._total_training_steps
            current_ratio = current_step / drop_path_burn_in_steps
            current_ratio = tf.minimum(1.0, current_ratio)
            drop_rate = current_ratio * drop_rate
        return drop_rate

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'cell_num': self._cell_num,
            'total_num_cells': self._total_num_cells,
            'total_training_steps': self._total_training_steps,
            'seed': self.seed
        }
        base_config = super(ScheduledDroppath, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConcreteDroppath(Layer):
    """Applies Concrete Droppath to the input.
        The Concrete Droppath layer randomly sets input path to 0 with a
        frequency considered as a weight of the layer optimized during training
        time, which helps prevent overfitting.
        Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
        all inputs is unchanged.
        Note that the Concrete Droppath layer only applies when `training` is set
        to True. When using `model.fit`, `training` will be appropriately set to
        True automatically, and in other contexts, you can set the kwarg explicitly
        to True when calling the layer. (This is in contrast to setting
        `trainable=False` for a Concrete Droppath layer. `trainable` does not affect
        the layer's behavior, as Dropout does not have any variables/weights that
        can be frozen during training.)
        Arguments:
            dropout_regularizer: A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$ with model precision
                $\tau$ (inverse observation noise) and N the number of
                instances in the dataset.
            init_min: dropout probability initializer min
            init_max: dropout probability initializer max
            seed: A Python integer to use as random seed.
        Call arguments:
            inputs: Input tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1,
                 seed=None, **kwargs):
        super(ConcreteDroppath, self).__init__(**kwargs)
        self.dropout_regularizer = dropout_regularizer
        self.init_min = tf.math.log(init_min) - tf.math.log(1. - init_min)
        self.init_max = tf.math.log(init_max) - tf.math.log(1. - init_max)
        self.p_logit = None
        self.seed = seed

    @tf.function
    def get_p(self):
        return tf.nn.sigmoid(self.p_logit[0])

    def build(self, input_shape=None):
        super(ConcreteDroppath, self).build(input_shape)
        # initialise p
        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.initializers.RandomUniform(self.init_min, self.init_max),
                                       trainable=True)
        # initialise regulariser / prior KL term
        input_dim = 1
        dropout_regularizer = self.get_p() * K.log(self.get_p())
        dropout_regularizer += (1. - self.get_p()) * K.log(1. - self.get_p())
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = dropout_regularizer
        self.add_loss(regularizer)

    def call(self, inputs, training=None):
        if training:
            return self.concrete_droppath(inputs)
        else:
            tf.identity(inputs)

    def concrete_droppath(self, x):
        """
        Concrete droppath - used at training and testing time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """

        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1
        unif_noise = tf.random.uniform(shape=[K.shape(x)[0], 1, 1, 1])
        drop_prob = (
            K.log(self.get_p() + eps)
            - K.log(1. - self.get_p() + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob
        retain_prob = 1. - self.get_p()
        x *= random_tensor
        x /= retain_prob
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_regularizer': self.dropout_regularizer,
            'init_min': self.init_min.numpy(),
            'init_max': self.init_max.numpy(),
            'p_logit': self.p_logit.numpy(),
            'seed': self.seed
        }
        base_config = super(ConcreteDroppath, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConcreteDropout(Layer):
    """Applies Concrete Dropout to the input.
        The Concrete Droppath layer randomly sets input path to 0 with a
        frequency considered as a weight of the layer optimized during training
        time, which helps prevent overfitting.
        Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
        all inputs is unchanged.
        Note that the Concrete Dropout layer only applies when `training` is set
        to True. When using `model.fit`, `training` will be appropriately set to
        True automatically, and in other contexts, you can set the kwarg explicitly
        to True when calling the layer. (This is in contrast to setting 
        `trainable=False` for a Concrete Dropout layer. `trainable` does not affect
        the layer's behavior, as Dropout does not have any variables/weights that
        can be frozen during training.)
        Arguments:
            dropout_regularizer: A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$ with model precision
                $\tau$ (inverse observation noise) and N the number of
                instances in the dataset.
            init_min: dropout probability initializer min
            init_max: dropout probability initializer max
            seed: A Python integer to use as random seed.
        Call arguments:
            inputs: Input tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1,
                 seed=None, **kwargs):
        super(ConcreteDropout, self).__init__(**kwargs)
        self.dropout_regularizer = dropout_regularizer
        self.init_min = tf.math.log(init_min) - tf.math.log(1. - init_min)
        self.init_max = tf.math.log(init_max) - tf.math.log(1. - init_max)
        self.p_logit = None
        self.seed = seed

    @tf.function
    def get_p(self):
        return tf.nn.sigmoid(self.p_logit[0])

    def build(self, input_shape=None):
        super(ConcreteDropout, self).build(input_shape)
        # initialise p
        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.initializers.RandomUniform(self.init_min, self.init_max),
                                       trainable=True)
        # initialise regulariser / prior KL term
        input_dim = np.prod(input_shape[1, 2])
        dropout_regularizer = self.get_p() * K.log(self.get_p())
        dropout_regularizer += (1. - self.get_p()) * K.log(1. - self.get_p())
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = dropout_regularizer
        self.add_loss(regularizer)

    def call(self, inputs, training=None):
        if training:
            return self.concrete_dropout(inputs)
        else:
            tf.identity(inputs)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time and testing time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=[K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], 1])
        drop_prob = (
            K.log(self.get_p() + eps)
            - K.log(1. - self.get_p() + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob
        retain_prob = 1. - self.get_p()
        x *= random_tensor
        x /= retain_prob
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_regularizer': self.dropout_regularizer,
            'init_min': self.init_min.numpy(),
            'init_max': self.init_max.numpy(),
            'p_logit': self.p_logit.numpy(),
            'seed': self.seed
        }
        base_config = super(ConcreteDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
