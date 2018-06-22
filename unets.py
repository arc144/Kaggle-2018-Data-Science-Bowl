import tensorflow as tf
import custom_Xception

slim = tf.contrib.slim


def vanilla(inputs, is_training=True, activation_fn='relu', padding='SAME',
            out_shape=[2], multi_head=False, scope='U-Net'):
    '''Vanilla U-Net archtecture according to the orignal paper'''
    assert padding in ['SAME', 'REFLECT', 'SYMMETRIC']

    with tf.variable_scope(scope) as sc:
        end_points_collections = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.max_pool2d,
                             slim.conv2d_transpose],
                            outputs_collections=[end_points_collections]), \
            slim.arg_scope([slim.conv2d,
                            slim.conv2d_transpose],
                           activation_fn=activation(activation_fn)), \
                slim.arg_scope([slim.batch_norm],
                               is_training=is_training):
            with tf.variable_scope('encoder'):
                net, bypass1 = down_conv_block(inputs, 64,
                                               conv_count=2,
                                               padding=padding,
                                               block=1)
                net, bypass2 = down_conv_block(net, 128,
                                               conv_count=2,
                                               padding=padding,
                                               block=2)
                net, bypass3 = down_conv_block(net, 256,
                                               conv_count=2,
                                               padding=padding,
                                               block=3)
                net, bypass4 = down_conv_block(net, 512,
                                               conv_count=2,
                                               padding=padding,
                                               block=4)
            with tf.variable_scope('transition'):
                net = pad_conv_2d(net, 1024, kernel=3, stride=1,
                                  padding=padding, scope='conv1')
                net = pad_conv_2d(net, 1024, kernel=3, stride=1,
                                  padding=padding, scope='conv2')
            with tf.variable_scope('decoder'):
                net = up_conv_block(net, bypass4, 512,
                                    conv_kernel=2,
                                    upconv_kernel=2,
                                    conv_count=2,
                                    padding=padding,
                                    block=4)
                net = up_conv_block(net, bypass3, 256,
                                    conv_kernel=2,
                                    upconv_kernel=2,
                                    conv_count=2,
                                    padding=padding,
                                    block=3)
                net = up_conv_block(net, bypass2, 128,
                                    conv_kernel=2,
                                    upconv_kernel=2,
                                    conv_count=2,
                                    padding=padding,
                                    block=2)
                net = up_conv_block(net, bypass1, 64,
                                    conv_kernel=2,
                                    upconv_kernel=2,
                                    conv_count=2,
                                    padding=padding,
                                    block=1)
            with tf.variable_scope('output'):
                full_mask_logits = slim.conv2d(
                    net, out_shape[0], [1, 1],
                    stride=1, padding='SAME', scope='conv1x1_full_mask')
                if multi_head:
                    borders_logits = slim.conv2d(
                        net, out_shape[1], [1, 1],
                        stride=1, padding='SAME', scope='conv1x1_borders')
                    logits = [full_mask_logits, borders_logits]
                else:
                    logits = [full_mask_logits]

            end_points = slim.utils.convert_collection_to_dict(
                end_points_collections)
            end_points['logits'] = logits
            return logits, end_points


def Xception_vanilla(inputs, is_training=True, activation_fn='relu', padding='SAME',
                     out_shape=[2], multi_head=False, scope='U-Net'):
    '''U-net using Xception as encoder
     Reference layers to add to decoder for input size=(256,256):
    bypass1: layer[0],   size=(256x256)
    bypass2: layer[11],   size=(128x128)
    bypass3: layer[21],  size=(64x64)
    bypass4: layer[31],  size=(32x32)
    bypass5: layer[121], size=(16x16)'''
    assert padding in ['SAME', 'REFLECT', 'SYMMETRIC']

    with tf.name_scope(scope) as sc:
        end_points_collections = sc + '_end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.max_pool2d,
                             slim.conv2d_transpose],
                            outputs_collections=[end_points_collections]), \
            slim.arg_scope([slim.conv2d,
                            slim.conv2d_transpose],
                           activation_fn=activation(activation_fn)), \
                slim.arg_scope([slim.batch_norm],
                               is_training=is_training):
            with tf.name_scope('encoder'):
                encoder = custom_Xception.Xception(
                    include_top=False, weights='imagenet',
                    input_tensor=inputs, pooling=None)
                bypass1 = encoder.layers[0].output
                bypass2 = encoder.layers[11].output
                bypass3 = encoder.layers[21].output
                bypass4 = encoder.layers[31].output
                bypass5 = encoder.layers[121].output
                net = encoder.output

            with tf.variable_scope('transition'):
                net = pad_conv_2d(net, 2048, kernel=3, stride=1,
                                  padding=padding, scope='conv1')

            with tf.variable_scope('decoder'):
                net = up_conv_block(net, bypass5, 1024,
                                    conv_count=1,
                                    conv_kernel=1,
                                    upconv_kernel=2,
                                    padding=padding,
                                    up_padding='SAME',
                                    block=5)
                net = up_conv_block(net, bypass4, 728,
                                    conv_count=1,
                                    conv_kernel=1,
                                    upconv_kernel=2,
                                    padding=padding,
                                    up_padding='SAME',
                                    block=4)
                net = up_conv_block(net, bypass3, 256,
                                    conv_count=1,
                                    conv_kernel=1,
                                    upconv_kernel=2,
                                    padding=padding,
                                    up_padding='SAME',
                                    block=3)
                net = up_conv_block(net, bypass2, 128,
                                    conv_count=1,
                                    conv_kernel=1,
                                    upconv_kernel=2,
                                    padding=padding,
                                    up_padding='SAME',
                                    block=2)
                net = up_conv_block(net, bypass1, 3,
                                    conv_count=1,
                                    conv_kernel=1,
                                    upconv_kernel=2,
                                    padding=padding,
                                    up_padding='SAME',
                                    block=1)
            with tf.variable_scope('output'):
                full_mask_logits = slim.conv2d(
                    net, out_shape[0], [1, 1],
                    stride=1, padding='SAME', scope='conv1x1_full_mask')
                if multi_head:
                    borders_logits = slim.conv2d(
                        net, out_shape[1], [1, 1],
                        stride=1, padding='SAME', scope='conv1x1_borders')
                    logits = [full_mask_logits, borders_logits]
                else:
                    logits = [full_mask_logits]
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collections)
                end_points['logits'] = logits

        return logits, encoder, end_points


def Xception_InceptionSE(inputs, is_training=True, activation_fn='relu', padding='SAME',
                         out_shape=[2], multi_head=False, scope='U-Net'):
    '''U-net using Xception as encoder
     Reference layers to add to decoder for input size=(256,256):
    bypass1: layer[0],   size=(256x256)
    bypass2: layer[11],   size=(128x128)
    bypass3: layer[21],  size=(64x64)
    bypass4: layer[31],  size=(32x32)
    bypass5: layer[121], size=(16x16)'''
    assert padding in ['SAME', 'REFLECT', 'SYMMETRIC']

    with tf.name_scope(scope) as sc:
        end_points_collections = sc + '_end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.max_pool2d,
                             slim.conv2d_transpose],
                            outputs_collections=[end_points_collections]), \
            slim.arg_scope([slim.conv2d,
                            slim.conv2d_transpose],
                           activation_fn=activation(activation_fn)), \
                slim.arg_scope([slim.batch_norm],
                               is_training=is_training):
            with tf.name_scope('encoder'):
                encoder = custom_Xception.Xception(
                    include_top=False, weights='imagenet',
                    input_tensor=inputs, pooling=None)
                bypass1 = encoder.layers[0].output
                bypass2 = encoder.layers[11].output
                bypass3 = encoder.layers[21].output
                bypass4 = encoder.layers[31].output
                bypass5 = encoder.layers[121].output
                net = encoder.output

            with tf.variable_scope('transition'):
                net = pad_conv_2d(net, 2048, kernel=3, stride=1,
                                  padding=padding, scope='conv1')

            with tf.variable_scope('decoder'):
                net = up_conv_block_v4(net, bypass5, 1024,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=5)
                net = up_conv_block_v4(net, bypass4, 728,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=4)
                net = up_conv_block_v4(net, bypass3, 256,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=3)
                net = up_conv_block_v4(net, bypass2, 128,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=2)
                net = up_conv_block_v3(net, bypass1, 3,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=1)
            with tf.variable_scope('output'):
                full_mask_logits = slim.conv2d(
                    net, out_shape[0], [1, 1],
                    stride=1, padding='SAME', scope='conv1x1_full_mask')
                if multi_head:
                    borders_logits = slim.conv2d(
                        net, out_shape[1], [1, 1],
                        stride=1, padding='SAME', scope='conv1x1_borders')
                    logits = [full_mask_logits, borders_logits]
                else:
                    logits = [full_mask_logits]
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collections)
                end_points['logits'] = logits

        return logits, encoder, end_points


def SE_Xception_vanillaSE(inputs, is_training=True, activation_fn='relu', padding='SAME',
                          out_shape=[2], multi_head=False, scope='U-Net'):
    '''U-net using SE_Xception as encoder
     Reference layers to add to decoder for input size=(256,256):
    bypass1: layer[0],   size=(256x256)
    bypass2: layer[21],   size=(128x128)
    bypass3: layer[36],  size=(64x64)
    bypass4: layer[51],  size=(32x32)
    bypass5: layer[186], size=(16x16)'''
    assert padding in ['SAME', 'REFLECT', 'SYMMETRIC']

    with tf.name_scope(scope) as sc:
        end_points_collections = sc + '_end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.max_pool2d,
                             slim.conv2d_transpose],
                            outputs_collections=[end_points_collections]), \
            slim.arg_scope([slim.conv2d,
                            slim.conv2d_transpose],
                           activation_fn=activation(activation_fn)), \
                slim.arg_scope([slim.batch_norm],
                               is_training=is_training):
            with tf.name_scope('encoder'):
                encoder = custom_Xception.SE_Xception(
                    include_top=False, weights='imagenet',
                    input_tensor=inputs, pooling=None)
                bypass1 = encoder.layers[0].output
                bypass2 = encoder.layers[21].output
                bypass3 = encoder.layers[36].output
                bypass4 = encoder.layers[51].output
                bypass5 = encoder.layers[186].output
                net = encoder.output

            with tf.variable_scope('transition'):
                net = pad_conv_2d(net, 2048, kernel=3, stride=1,
                                  padding=padding, scope='conv1')

            with tf.variable_scope('decoder'):
                net = up_conv_block_v3(net, bypass5, 1024,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=5)
                net = up_conv_block_v3(net, bypass4, 728,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=4)
                net = up_conv_block_v3(net, bypass3, 256,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=3)
                net = up_conv_block_v3(net, bypass2, 128,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=2)
                net = up_conv_block_v3(net, bypass1, 3,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=1)
            with tf.variable_scope('output'):
                full_mask_logits = slim.conv2d(
                    net, out_shape[0], [1, 1],
                    stride=1, padding='SAME', scope='conv1x1_full_mask')
                if multi_head:
                    borders_logits = slim.conv2d(
                        net, out_shape[1], [1, 1],
                        stride=1, padding='SAME', scope='conv1x1_borders')
                    logits = [full_mask_logits, borders_logits]
                else:
                    logits = [full_mask_logits]
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collections)
                end_points['logits'] = logits

        return logits, encoder, end_points


def InceptionResNetV2_vanilla(inputs, is_training=True, activation_fn='relu', padding='SAME',
                              out_shape=[2], multi_head=False, scope='U-Net'):  # TO DO
    '''U-net using InceptionResNetV2 as encoder
    Reference layers to add to decoder for input size=(256,256):
    bypass1: layer[0],   size=(256x256)
    bypass2: layer[11],   size=(128x128)
    bypass3: layer[21],  size=(64x64)
    bypass4: layer[31],  size=(32x32)
    bypass5: layer[121], size=(16x16)'''
    assert padding in ['SAME', 'REFLECT', 'SYMMETRIC']

    with tf.name_scope(scope) as sc:
        end_points_collections = sc + '_end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.max_pool2d,
                             slim.conv2d_transpose],
                            outputs_collections=[end_points_collections]), \
            slim.arg_scope([slim.conv2d,
                            slim.conv2d_transpose],
                           activation_fn=activation(activation_fn)), \
                slim.arg_scope([slim.batch_norm],
                               is_training=is_training):
            with tf.name_scope('encoder'):
                encoder = custom_Xception.Xception(
                    include_top=False, weights='imagenet',
                    input_tensor=inputs, pooling=None)
                bypass1 = encoder.layers[0].output
                bypass2 = encoder.layers[11].output
                bypass3 = encoder.layers[21].output
                bypass4 = encoder.layers[31].output
                bypass5 = encoder.layers[121].output
                net = encoder.output

            with tf.variable_scope('transition'):
                net = pad_conv_2d(net, 2048, kernel=3, stride=1,
                                  padding=padding, scope='conv1')

            with tf.variable_scope('decoder'):
                net = up_conv_block_v3(net, bypass5, 1024,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=5)
                net = up_conv_block_v3(net, bypass4, 728,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=4)
                net = up_conv_block_v3(net, bypass3, 256,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=3)
                net = up_conv_block_v3(net, bypass2, 128,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=2)
                net = up_conv_block_v3(net, bypass1, 3,
                                       conv_count=1,
                                       conv_kernel=3,
                                       upconv_kernel=2,
                                       padding=padding,
                                       up_padding='SAME',
                                       block=1)
            with tf.variable_scope('output'):
                full_mask_logits = slim.conv2d(
                    net, out_shape[0], [1, 1],
                    stride=1, padding='SAME', scope='conv1x1_full_mask')
                if multi_head:
                    borders_logits = slim.conv2d(
                        net, out_shape[1], [1, 1],
                        stride=1, padding='SAME', scope='conv1x1_borders')
                    logits = [full_mask_logits, borders_logits]
                else:
                    logits = [full_mask_logits]
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collections)
                end_points['logits'] = logits

        return logits, encoder, end_points


def down_conv_block(inputs, out_channels, conv_count=2, padding='SAME',
                    bn=True, block=0):
    '''Convolution block (encoder part)'''
    tmp = inputs
    with tf.variable_scope('block{}'.format(block)):
        for i in range(conv_count):
            tmp = pad_conv_2d(tmp, out_channels,
                              kernel=3, stride=1,
                              padding=padding,
                              scope='conv{}'.format(i),
                              bn=bn)
        conv = tmp  # Get conv reference to be used as bypass
        tmp = slim.max_pool2d(tmp, [3, 3], stride=2,
                              padding='SAME', scope='pool')
    return tmp, conv


def up_conv_block(inputs, bypass, out_channels, upconv_kernel=2,
                  conv_kernel=3, conv_count=2,
                  padding='SAME', up_padding='SAME',
                  bn=True, block=0):
    '''Convolution block (decoder part)'''
    tmp = inputs
    with tf.variable_scope('block{}'.format(block)):
        tmp = slim.conv2d_transpose(tmp, out_channels,
                                    [upconv_kernel, upconv_kernel],
                                    stride=2,
                                    padding=up_padding,
                                    scope='up_conv')
        tmp = tf.concat([tmp, bypass], axis=-1, name='concat')
        for i in range(conv_count):
            tmp = pad_conv_2d(tmp, out_channels,
                              kernel=conv_kernel, stride=1,
                              padding=padding,
                              scope='conv{}'.format(i),
                              bn=bn)
    return tmp


def up_conv_block_v2(inputs, bypass, out_channels, upconv_kernel=2,
                     conv_kernel=3,
                     padding='SAME', up_padding='SAME',
                     bn=True, block=0):
    '''Convolution block (decoder part) using 2 convs with different channels
       First conv doubles the channels and then second shrinks it back'''
    tmp = inputs
    with tf.variable_scope('block{}'.format(block)):
        tmp = slim.conv2d_transpose(tmp, out_channels,
                                    [upconv_kernel, upconv_kernel],
                                    stride=2,
                                    padding=up_padding,
                                    scope='up_conv')
        tmp = tf.concat([tmp, bypass], axis=-1, name='concat')
        tmp = pad_conv_2d(tmp, 2 * out_channels,
                          kernel=conv_kernel, stride=1,
                          padding=padding,
                          scope='conv1',
                          bn=bn)
        tmp = pad_conv_2d(tmp, out_channels,
                          kernel=conv_kernel, stride=1,
                          padding=padding,
                          scope='conv2',
                          bn=bn)
    return tmp


def up_conv_block_v3(inputs, bypass, out_channels,
                     upconv_kernel=2,
                     conv_kernel=3, conv_count=2,
                     padding='SAME', up_padding='SAME',
                     se_reduction_ratio=16,
                     bn=True, block=0):
    '''Convolution block (decoder part) with addition to SE block at the end.
       Squeeze and excitation block is believed to improve model since it
       weights the importance of each channel, which are fundamental to up_block'''
    tmp = inputs
    with tf.variable_scope('block{}'.format(block)):
        tmp = slim.conv2d_transpose(tmp, out_channels,
                                    [upconv_kernel, upconv_kernel],
                                    stride=2,
                                    padding=up_padding,
                                    scope='up_conv')
        tmp = tf.concat([tmp, bypass], axis=-1, name='concat')
        for i in range(conv_count):
            tmp = pad_conv_2d(tmp, out_channels,
                              kernel=conv_kernel, stride=1,
                              padding=padding,
                              scope='conv{}'.format(i),
                              bn=bn)
        tmp = squeeze_excitation_block(
            tmp, out_channels, ratio=se_reduction_ratio)

    return tmp


def up_conv_block_v4(inputs, bypass, out_channels,
                     upconv_kernel=2,
                     padding='SAME', up_padding='SAME',
                     inception_ratios=[4, 4, 4, 4],
                     se_reduction_ratio=16,
                     bn=True, se=True, block=0):
    '''Convolution block (decoder part) with addition Inception idea'''
    tmp = inputs
    with tf.variable_scope('block{}'.format(block)):
        tmp = slim.conv2d_transpose(tmp, out_channels,
                                    [upconv_kernel, upconv_kernel],
                                    stride=2,
                                    padding=up_padding,
                                    scope='up_conv')
        tmp = tf.concat([tmp, bypass], axis=-1, name='concat')

        tmp = inception_block(
            tmp, out_channels, padding=padding, ratios=inception_ratios)
        if se:
            tmp = squeeze_excitation_block(
                tmp, out_channels, ratio=se_reduction_ratio)

    return tmp


def pad_conv_2d(inputs, out_channels, kernel, stride, padding, scope, bn=True):
    '''Conv2d fun to deal with paddings'''
    if padding == 'SAME' or padding == 'VALID':
        conv = slim.conv2d(inputs, out_channels,
                           [kernel, kernel], stride=stride, padding=padding,
                           scope=scope)
    else:
        if kernel % 2 == 0:
            p = int(kernel / 2)
            pad_tf = tf.constant([[0, 0],
                                  [0, p],
                                  [0, p],
                                  [0, 0]])
        else:
            p = int((kernel - 1) / 2)
            pad_tf = tf.constant([[0, 0],
                                  [p, p],
                                  [p, p],
                                  [0, 0]])
        inputs = tf.pad(inputs, pad_tf, padding)
        conv = slim.conv2d(inputs, out_channels,
                           [kernel, kernel], stride=stride, padding='VALID',
                           scope=scope)
        if bn:
            conv = slim.batch_norm(conv)

    return conv


def squeeze_excitation_block(inputs, out_channels, ratio=16):
    '''Fundamental block of SENets, winner of 2017 Imagenet'''
    with tf.name_scope('se_block'):
        with tf.name_scope('squeeze'):
            tmp = tf.reduce_mean(
                inputs, axis=[1, 2], name='global_avg_pool')

        with tf.name_scope('excitation'):
            tmp = slim.fully_connected(
                tmp,
                num_outputs=out_channels // ratio,
                activation_fn=tf.nn.relu,
                scope='fc1')

            tmp = slim.fully_connected(
                tmp,
                num_outputs=out_channels,
                activation_fn=tf.nn.sigmoid,
                scope='fc2')

        with tf.name_scope('scale'):
            tmp = tf.reshape(tmp, [-1, 1, 1, out_channels])
            tmp = tmp * inputs

    return tmp


def inception_block(inputs, out_channels, padding, ratios=[4, 4, 4, 4]):
    with tf.name_scope('inception_block'):
        conv1 = pad_conv_2d(inputs, out_channels // ratios[0],
                            kernel=1, stride=1,
                            padding=padding,
                            scope='branch1x1')
        conv3 = conv1_convx(inputs, out_channels // ratios[1],
                            kernel=3, stride=1,
                            padding=padding,
                            scope='branch3x3')
        conv5 = conv1_convx(inputs, out_channels // ratios[2],
                            kernel=3, stride=1,
                            conv_count=2,
                            padding=padding,
                            scope='branch5x5')
        pool = slim.max_pool2d(inputs, [3, 3], stride=1,
                               padding='SAME', scope='branch_pool')
        pool = pad_conv_2d(pool, out_channels // ratios[3],
                           kernel=1, stride=1,
                           padding=padding,
                           scope='branch_pool_conv1x1')
        out = tf.concat([conv1, conv3, conv5, pool], axis=-1, name='concat')
    return out


def conv1_convx(inputs, out_channels, kernel, stride=1, conv_count=1, ratio=2, padding='SAME', scope=None):
    '''ConvX preceeded of a conv1x1 with out/ratio feature maps'''
    with tf.variable_scope(scope):
        conv = pad_conv_2d(inputs, out_channels // ratio,
                           kernel=1, stride=stride,
                           padding=padding, scope='conv1x1')
        for i in range(conv_count):
            conv = pad_conv_2d(conv, out_channels,
                               kernel=kernel, stride=stride,
                               padding=padding,
                               scope='conv{0}x{0}_{1}'.format(scope, i))
    return conv


def activation(act_fun):
    """ Activation function. """
    if act_fun == 'elu':
        a = tf.nn.elu

    elif act_fun == 'leaky_relu':
        a = leaky_relu

    elif act_fun == 'relu':
        a = tf.nn.relu

    elif act_fun == 'selu':
        a = tf.nn.selu

    return a


def leaky_relu(self, z, name=None):
    """Leaky ReLU."""
    return tf.maximum(0.01 * z, z, name=name)
