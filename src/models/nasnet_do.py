# -*- coding: utf-8 -*-
"""NASNet-A models for Keras.
NASNet refers to Neural Architecture Search Network, a family of models
that were designed automatically by learning the model architectures
directly on the dataset of interest.
Here we consider NASNet-A, the highest performance model that was found
for the CIFAR-10 dataset, and then extended to ImageNet 2012 dataset,
obtaining state of the art performance on CIFAR-10 and ImageNet 2012.
Only the NASNet-A models, and their respective weights, which are suited
for ImageNet 2012 are provided.
The below table describes the performance on ImageNet 2012:
--------------------------------------------------------------------------------
      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)
--------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.0 %  |   91.6 %  |       564 M    |     5.3    |
|   NASNet-A (6 @ 4032)  |   82.7 %  |   96.2 %  |      23.8 B    |    88.9    |
--------------------------------------------------------------------------------
Reference:
  - [Learning Transferable Architectures for Scalable Image Recognition](
      https://arxiv.org/abs/1707.07012) (CVPR 2018)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from models.nasnet_utils_do import ScheduledDropout, ScheduledDroppath, ConcreteDropout, ConcreteDroppath
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.keras.applications.imagenet_utils import correct_pad

from . import NetType

TF_NASNET_LARGE_WEIGHT_PATH = 'https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-large.h5'
TF_NASNET_LARGE_WEIGHT_PATH_NO_TOP = 'https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-large-no-top.h5'


def NASNet_large_do(net_type, include_top=True, do_rate=0.3, weights='imagenet', input_tensor=None,
                    input_shape=None, total_training_steps=None, penultimate_filters=4032, num_blocks=6,
                    stem_block_filters=96, skip_reduction=True, filter_multiplier=2, pooling=None, classes=1000):
    """Instantiates a NASNet model.

    Optionally loads weights pre-trained
    on ImageNet. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    You should set `image_data_format='channels_last'` in your Keras config
    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 331x331.

    # Arguments
        net_type: type of dropout or droppath to use in the network
            specified in models.__init__.py.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        do_rate: dropout or droppath rate (if needed).
        weights: `None` (random initialization) or
          path to weights file which meets network requirements or
          `imagenet` (ImageNet weights)
          For loading `imagenet` weights, `input_shape` should be (331, 331, 3)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(331, 331, 3)` for NASNetLarge.
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 32.
          E.g. `(224, 224, 3)` would be one valid value.
        total_training_steps: number of total steps performed during training.
        penultimate_filters: number of filters in the penultimate layer.
          NASNet models use the notation `NASNet (N @ P)`, where:
          -   N is the number of blocks
          -   P is the number of penultimate filters
        num_blocks: number of repeated blocks of the NASNet model.
          NASNet models use the notation `NASNet (N @ P)`, where:
          -   N is the number of blocks
          -   P is the number of penultimate filters
        stem_block_filters: number of filters in the initial stem block
        skip_reduction: whether to skip the reduction step at the tail
          end of the network.
        filter_multiplier:controls the width of the network.
          - If `filter_multiplier` < 1.0, proportionally decreases the number
            of filters in each layer.
          - If `filter_multiplier` > 1.0, proportionally increases the number
            of filters in each layer.
          - If `filter_multiplier` = 1, default number of filters from the
            paper are used at each layer.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Xception model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The NASNet model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=331,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=False,
                                      weights=None)  # weights=None to prevent input channels equality check

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if penultimate_filters % (24 * (filter_multiplier ** 2)) != 0:
        raise ValueError(
            'For NASNet-A models, the `penultimate_filters` must be a multiple '
            'of 24 * (`filter_multiplier` ** 2). Current value: %d' %
            penultimate_filters)
    filters = penultimate_filters // 24

    x = Conv2D(stem_block_filters, (3, 3), strides=(2, 2), padding="same",
               use_bias=False, name='stem_conv1', kernel_initializer='he_normal')(img_input)
    x = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)
    #conv1
    if net_type == NetType.mc_do:
        x = Dropout(do_rate, name='dropout')(x, training=True)
    elif net_type == NetType.mc_df:
        x = Dropout(do_rate, noise_shape=(x.shape[0], 1, 1, x.shape[-1]), name='dropfilter')(x, training=True)

    total_num_cells = 4 + 3 * num_blocks
    cell_counter = 0
    p = None
    x, p = _reduction_a_cell_do(x, p, filters // (filter_multiplier ** 2), net_type=net_type, cell_num=cell_counter,
                                total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                do_rate=do_rate, block_id='stem_1')
    cell_counter += 1
    #conv2
    x, p = _reduction_a_cell_do(x, p, filters // filter_multiplier, net_type=net_type, cell_num=cell_counter,
                                total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                do_rate=do_rate, block_id='stem_2')
    cell_counter += 1

    for i in range(num_blocks):
        x, p = _normal_a_cell_do(x, p, filters, net_type=net_type, cell_num=cell_counter,
                                 total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                 do_rate=do_rate, block_id='%d' % (i))
        cell_counter += 1
    #conv3
    x, p0 = _reduction_a_cell_do(x, p, filters * filter_multiplier, net_type=net_type, cell_num=cell_counter,
                                 total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                 do_rate=do_rate, block_id='reduce_%d' % (num_blocks))
    cell_counter += 1

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell_do(x, p, filters * filter_multiplier, net_type=net_type, cell_num=cell_counter,
                                 total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                 do_rate=do_rate, block_id='%d' % (num_blocks + i + 1))
        cell_counter += 1
    #conv4
    x, p0 = _reduction_a_cell_do(x, p, filters * filter_multiplier ** 2, net_type=net_type, cell_num=cell_counter,
                                 total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                 do_rate=do_rate, block_id='reduce_%d' % (2 * num_blocks))
    cell_counter += 1
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell_do(x, p, filters * filter_multiplier ** 2, net_type=net_type, cell_num=cell_counter,
                                 total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                 do_rate=do_rate, block_id='%d' % (2 * num_blocks + i + 1))
        cell_counter += 1
    #conv5
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='NASNet')
    # Create donor model
    if input_shape[-1] > 3 and weights is not None:
        donor_input_shape = (*input_shape[:-1], 3)
        donor_model = get_donor_model(include_top, input_tensor=None,
                                      input_shape=donor_input_shape,
                                      penultimate_filters=penultimate_filters,
                                      num_blocks=num_blocks,
                                      stem_block_filters=stem_block_filters,
                                      skip_reduction=skip_reduction,
                                      pooling=pooling,
                                      classes=classes)

    # load weights
    if weights is not None and input_shape[-1] > 3:
        if weights == 'imagenet':
            if include_top:
                print('Loading pretrained ImageNet weights, include top for NASNet backbone')
                weights_path = get_file('nasnet_large.h5',
                                        TF_NASNET_LARGE_WEIGHT_PATH,
                                        cache_subdir='models',
                                        file_hash='11577c9a518f0070763c2b964a382f17')
            else:
                print('Loading pretrained ImageNet weights, exclude top for NASNet backbone')
                weights_path = get_file('nasnet_large_no_top.h5',
                                        TF_NASNET_LARGE_WEIGHT_PATH_NO_TOP,
                                        cache_subdir='models',
                                        file_hash='d81d89dc07e6e56530c4e77faddd61b5')
        else:
            print('Parameter "pretrained_weights" is expected to be "imagenet". However you can pass path to weights '
                  'if you are sure about what you are doing!')
            if os.path.exists(weights):
                weights_path = weights
            else:
                print('Parameter "pretrained_weights" is expected to be "imagenet" or path to weights. Considered to '
                      f'be path, but it doesn\'t exist: {weights}')
        if input_shape[-1] > 3:
            print(
                f'Copying pretrained weights to model with {input_shape[-1]} input channels for NASNet backbone')
            donor_model.load_weights(weights_path)

            donor_model_layers_weights = [d_l for d_l in donor_model.layers if len(d_l.weights) > 0]
            j = 0
            already_copied_layers = []
            for i, l in enumerate([l for l in model.layers if len(l.weights) > 0]):
                if j >= len(donor_model_layers_weights):
                    break
                while j in already_copied_layers:
                    j += 1
                d_l = donor_model_layers_weights[j]
                # dropout in target model - skip to next layer
                if 'dropout' in l.name and 'dropout' not in d_l.name or \
                        'droppath' in l.name and 'droppath' not in d_l.name or \
                        'dropfilter' in l.name and 'dropfilter' not in d_l.name:
                    continue
                # first weighted layer in target model - adding weights for uncovered channels
                if i == 0:
                    new_w = tf.tile(d_l.weights[0], (1, 1, 2, 1))[:, :, :input_shape[-1], :]
                    l.weights[0].assign(new_w)
                    j += 1
                # layers names are identical - copy weights
                elif l.name == d_l.name:
                    for (w, d_w) in zip(l.weights, d_l.weights):
                        w.assign(d_w)
                    j += 1
                # layer order is broken - search for the matching donor layer and copy weights
                else:
                    for k in range(j+1, len(donor_model_layers_weights)):
                        d_l_next = donor_model_layers_weights[k]
                        if l.name == d_l_next.name:
                            for (w, d_n_w) in zip(l.weights, d_l_next.weights):
                                w.assign(d_n_w)
                            already_copied_layers.append(k)
                            break
                        if k == len(donor_model_layers_weights) -1:
                            raise ValueError
            assert j == len(donor_model_layers_weights)
            del donor_model
        else:
            model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print('No pretrained weights passed')

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def _separable_conv_block_do(ip, filters, net_type, kernel_size=(3, 3), strides=(1, 1), do_rate=None, cell_num=None,
                             total_num_cells=None, total_training_steps=None, block_id=None):
    """Adds 2 blocks of [relu-separable conv-batchnorm].

    Args:
        ip: Input tensor
        filters: Number of output filters per layer
        net_type: type of dropout or droppath to use in the network
            specified in models.__init__.py.
        kernel_size: Kernel size of separable convolutions
        strides: Strided convolution for downsampling
        do_rate: Dropout or droppath rate
        cell_num: Cell number in the network
        total_num_cells: Number of cells in the network
        total_training_steps: Number of total steps performed during training
        block_id: String block_id
    Returns:
        A Keras tensor
    """

    with K.name_scope('separable_conv_block_%s' % block_id):
        x = Activation('relu')(ip)
        if strides == (2, 2):
            x = ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                     name='separable_conv_1_pad_%s' % block_id)(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % block_id,
                                   padding=conv_pad, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='separable_conv_1_bn_%s' % (block_id))(x)
        if net_type == NetType.mc_do:
            x = Dropout(do_rate, name='dropout0_%s' % (block_id))(x, training=True)
        elif net_type == NetType.mc_df:
            x = Dropout(do_rate, noise_shape=(x.shape[0], 1, 1, x.shape[-1]), name='dropfilter_%s' % (block_id))\
                (x, training=True)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % block_id, padding='same',
                                   use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='separable_conv_2_bn_%s' % (block_id))(x)
        if net_type == NetType.mc_do:
            x = Dropout(do_rate, name='dropout1_%s' % (block_id))(x, training=True)
        elif net_type == NetType.mc_df:
            x = Dropout(do_rate, noise_shape=(x.shape[0], 1, 1, x.shape[-1]), name='dropfilter_%s' % (block_id))\
                (x, training=True)
        x = Activation('relu')(x)

        if net_type == NetType.sdo:
            if cell_num is None or total_num_cells is None:
                raise ValueError('Please specify cell number for correct Scheduled MC dropout')
            x = ScheduledDropout(do_rate, cell_num=cell_num, total_num_cells=total_num_cells,
                                 total_training_steps=total_training_steps, name='scheduled_dropout_%s' % (block_id))\
                (x, training=True)
        if net_type == NetType.sdp:
            if cell_num is None or total_num_cells is None:
                raise ValueError('Please specify cell number for correct Scheduled MC droppath')
            x = ScheduledDroppath(do_rate, cell_num=cell_num, total_num_cells=total_num_cells,
                                 total_training_steps=total_training_steps, name='scheduled_droppath_%s' % (block_id))\
                (x, training=True)
        elif net_type == NetType.cdo:
            x = ConcreteDropout(name='concrete_dropout_%s' % (block_id))(x, training=True)
        elif net_type == NetType.cdp:
            x = ConcreteDroppath(name='concrete_droppath_%s' % (block_id))(x, training=True)

    return x


def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1), block_id=None):
    """Adds 2 blocks of [relu-separable conv-batchnorm].

    Args:
        ip: Input tensor
        filters: Number of output filters per layer
        kernel_size: Kernel size of separable convolutions
        strides: Strided convolution for downsampling
        block_id: String block_id
    Returns:
        A Keras tensor
    """

    with K.name_scope('separable_conv_block_%s' % block_id):
        x = Activation('relu')(ip)
        if strides == (2, 2):
            x = ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                     name='separable_conv_1_pad_%s' % block_id)(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % block_id,
                                   padding=conv_pad, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='separable_conv_1_bn_%s' % (block_id))(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % block_id, padding='same',
                                   use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='separable_conv_2_bn_%s' % (block_id))(x)
        x = Activation('relu')(x)
    return x


def _adjust_block(p, ip, filters, block_id=None):
    """Adjusts the input `previous path` to match the shape of the `input`.
    Used in situations where the output number of filters needs to be changed.

    Args:
        p: Input tensor which needs to be modified
        ip: Input tensor whose shape needs to be matched
        filters: Number of output filters to be matched
        block_id: String block_id
    Returns:
        Adjusted Keras tensor
    """

    ip_shape = K.int_shape(ip)

    if p is not None:
        p_shape = K.int_shape(p)

    with K.name_scope('adjust_block'):
        if p is None:
            p = ip

        elif p_shape[-2] != ip_shape[-2]:
            with K.name_scope('adjust_reduction_block_%s' % block_id):
                p = Activation('relu', name='adjust_relu_1_%s' % block_id)(p)
                p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid',
                                             name='adjust_avg_pool_1_%s' % block_id)(p)
                p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False,
                                   name='adjust_conv_1_%s' % block_id, kernel_initializer='he_normal')(p1)
                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D((1, 1),
                                             strides=(2, 2),
                                             padding='valid',
                                             name='adjust_avg_pool_2_%s' % block_id)(
                                                 p2)
                p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False,
                                   name='adjust_conv_2_%s' % block_id, kernel_initializer='he_normal')(p2)
                p = layers.concatenate([p1, p2])
                p = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='adjust_bn_%s' % block_id)(p)

        elif p_shape[-1] != filters:
            with K.name_scope('adjust_projection_block_%s' % block_id):
                p = Activation('relu')(p)
                p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                                  name='adjust_conv_projection_%s' % block_id, use_bias=False,
                                  kernel_initializer='he_normal')(p)
                p = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, name='adjust_bn_%s' % block_id)(p)
    return p


def _normal_a_cell_do(ip, p, filters, net_type, cell_num, total_num_cells,
                      total_training_steps, do_rate=0.3, block_id=None):
    """Adds a Normal cell for NASNet-A (Fig. 4 in the paper).

    Args:
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        net_type: Type of dropout or droppath to use in the network
            specified in models.__init__.py.
        cell_num: Cell Number in the network
        total_num_cells: Number of cells in the network
        total_training_steps: Number of total steps performed during training
        do_rate: Dropout or droppath rate
        block_id: String block_id

    Returns:
        A Keras tensor
    """

    with K.name_scope('normal_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % block_id,
                          use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='normal_bn_1_%s' % block_id)(h)
        if net_type == NetType.mc_do:
            h = Dropout(do_rate, name='dropout_%s' % (block_id))(h, training=True)
        elif net_type == NetType.mc_df:
            h = Dropout(do_rate, noise_shape=(h.shape[0], 1, 1, h.shape[-1]), name='dropfilter_%s' % (block_id))\
                (h, training=True)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block_do(h, filters, net_type, kernel_size=(5, 5), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='normal_left1_%s' % block_id)
            x1_2 = _separable_conv_block_do(p, filters, net_type, do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='normal_right1_%s' % block_id)
            x1 = layers.add([x1_1, x1_2], name='normal_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block_do(p, filters, net_type, (5, 5), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='normal_left2_%s' % block_id)
            x2_2 = _separable_conv_block_do(p, filters, net_type, (3, 3), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='normal_right2_%s' % block_id)
            x2 = layers.add([x2_1, x2_2], name='normal_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left3_%s' % block_id)(h)

            p_add = p
            x3 = layers.add([x3, p_add], name='normal_add_3_%s' % block_id)

        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % block_id)(p)
            x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                    name='normal_right4_%s' % block_id)(p)
            x4 = layers.add([x4_1, x4_2], name='normal_add_4_%s' % block_id)

        with K.name_scope('block_5'):
            x5 = _separable_conv_block_do(h, filters, net_type, do_rate=do_rate, cell_num=cell_num,
                                          total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                          block_id='normal_left5_%s' % block_id)
            x5 = layers.add([x5, h], name='normal_add_5_%s' % block_id)

        x = layers.concatenate([p, x1, x2, x3, x4, x5], name='normal_concat_%s' % block_id)
    return x, ip


def _normal_a_cell(ip, p, filters, block_id=None):
    """Adds a Normal cell for NASNet-A (Fig. 4 in the paper).

    Args:
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id

    Returns:
        A Keras tensor
    """

    with K.name_scope('normal_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % block_id,
                          use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='normal_bn_1_%s' % block_id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5), block_id='normal_left1_%s' % block_id)
            x1_2 = _separable_conv_block(p, filters, block_id='normal_right1_%s' % block_id)
            x1 = layers.add([x1_1, x1_2], name='normal_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5, 5), block_id='normal_left2_%s' % block_id)
            x2_2 = _separable_conv_block(p, filters, (3, 3), block_id='normal_right2_%s' % block_id)
            x2 = layers.add([x2_1, x2_2], name='normal_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left3_%s' % block_id)(h)
            x3 = layers.add([x3, p], name='normal_add_3_%s' % block_id)

        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % block_id)(p)
            x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                    name='normal_right4_%s' % block_id)(p)
            x4 = layers.add([x4_1, x4_2], name='normal_add_4_%s' % block_id)

        with K.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, block_id='normal_left5_%s' % block_id)
            x5 = layers.add([x5, h], name='normal_add_5_%s' % block_id)

        x = layers.concatenate([p, x1, x2, x3, x4, x5], name='normal_concat_%s' % block_id)
    return x, ip


def _reduction_a_cell_do(ip, p, filters, net_type, cell_num, total_num_cells, total_training_steps, do_rate=0.3,
                         block_id=None):
    """Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).

    Args:
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        net_type: Type of dropout or droppath to use in the network
            specified in models.__init__.py.
        cell_num: Cell Number in the network
        total_num_cells: Number of cells in the network
        total_training_steps: Number of total steps performed during training
        do_rate: Dropout or droppath rate
        block_id: String block_id

    Returns:
        A Keras tensor
    """

    with K.name_scope('reduction_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % block_id,
                   use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='reduction_bn_1_%s' % block_id)(h)
        if net_type == NetType.mc_do:
            h = Dropout(do_rate, name='dropout_%s' % (block_id))(h, training=True)
        elif net_type == NetType.mc_df:
            h = Dropout(do_rate, noise_shape=(h.shape[0], 1, 1, h.shape[-1]), name='dropfilter_%s' % (block_id))\
                (h, training=True)
        h3 = ZeroPadding2D(padding=correct_pad(h, 3), name='reduction_pad_1_%s' % block_id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block_do(h, filters, net_type, (5, 5), strides=(2, 2), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='reduction_left1_%s' % block_id)
            x1_2 = _separable_conv_block_do(p, filters, net_type, (7, 7), strides=(2, 2), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='reduction_right1_%s' % block_id)
            x1 = layers.add([x1_1, x1_2], name='reduction_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='reduction_left2_%s' % block_id)(h3)
            x2_2 = _separable_conv_block_do(p, filters, net_type, (7, 7), strides=(2, 2), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='reduction_right2_%s' % block_id)
            x2 = layers.add([x2_1, x2_2], name='reduction_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='valid', name='reduction_left3_%s' % block_id)(h3)
            x3_2 = _separable_conv_block_do(p, filters, net_type, (5, 5), strides=(2, 2), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='reduction_right3_%s' % block_id)
            x3 = layers.add([x3_1, x3_2], name='reduction_add3_%s' % block_id)

        with K.name_scope('block_4'):
            x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='reduction_left4_%s' % block_id)(x1)
            x2_add = x2
            x4 = layers.add([x2_add, x4])

        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block_do(x1, filters, net_type, (3, 3), do_rate=do_rate, cell_num=cell_num,
                                            total_num_cells=total_num_cells, total_training_steps=total_training_steps,
                                            block_id='reduction_left4_%s' % block_id)
            x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid',
                                name='reduction_right5_%s' % block_id)(h3)

            x5 = layers.add([x5_1, x5_2], name='reduction_add4_%s' % block_id)

        x = layers.concatenate([x2, x3, x4, x5], name='reduction_concat_%s' % block_id)
        return x, ip


def _reduction_a_cell(ip, p, filters, block_id=None):
    """Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).

    Args:
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id

    Returns:
        A Keras tensor
    """

    with K.name_scope('reduction_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % block_id,
                   use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='reduction_bn_1_%s' % block_id)(h)
        h3 = ZeroPadding2D(padding=correct_pad(h, 3), name='reduction_pad_1_%s' % block_id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2),
                                            block_id='reduction_left1_%s' % block_id)
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2),
                                            block_id='reduction_right1_%s' % block_id)
            x1 = layers.add([x1_1, x1_2], name='reduction_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='reduction_left2_%s' % block_id)(h3)
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2),
                                            block_id='reduction_right2_%s' % block_id)
            x2 = layers.add([x2_1, x2_2], name='reduction_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='valid',
                                           name='reduction_left3_%s' % block_id)(h3)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2),
                                            block_id='reduction_right3_%s' % block_id)
            x3 = layers.add([x3_1, x3_2], name='reduction_add3_%s' % block_id)

        with K.name_scope('block_4'):
            x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                         name='reduction_left4_%s' % block_id)(x1)
            x4 = layers.add([x2, x4])

        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3, 3),
                                            block_id='reduction_left4_%s' % block_id)
            x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid',
                                name='reduction_right5_%s' % block_id)(h3)
            x5 = layers.add([x5_1, x5_2], name='reduction_add4_%s' % block_id)

        x = layers.concatenate([x2, x3, x4, x5], name='reduction_concat_%s' % block_id)
        return x, ip


def get_donor_model(include_top=True, input_tensor=None, input_shape=None, penultimate_filters=4032,
                    num_blocks=6, stem_block_filters=96, skip_reduction=True, filter_multiplier=2,
                    pooling=None, classes=1000):

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=331,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=False,
                                      weights=None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if penultimate_filters % (24 * (filter_multiplier ** 2)) != 0:
        raise ValueError(
            'For NASNet-A models, the `penultimate_filters` must be a multiple '
            'of 24 * (`filter_multiplier` ** 2). Current value: %d' %
            penultimate_filters)
    filters = penultimate_filters // 24

    x = Conv2D(stem_block_filters, (3, 3), strides=(2, 2), padding="valid",
               use_bias=False, name='stem_conv1', kernel_initializer='he_normal')(img_input)
    x = BatchNormalization(momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)

    p = None
    x, p = _reduction_a_cell(x, p, filters // (filter_multiplier ** 2), block_id='stem_1')
    x, p = _reduction_a_cell(x, p, filters // filter_multiplier, block_id='stem_2')

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters, block_id='%d' % (i))

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier, block_id='reduce_%d' % (num_blocks))

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier, block_id='%d' % (num_blocks + i + 1))

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier ** 2, block_id='reduce_%d' % (2 * num_blocks))
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier ** 2, block_id='%d' % (2 * num_blocks + i + 1))

    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return Model(inputs, x, name='donor_NASNet')


if __name__ == '__main__':
    NASNet_large_do(net_type=NetType.mc_do, include_top=False, input_shape=(256, 256, 3)).summary()
