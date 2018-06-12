import tensorflow as tf

slim = tf.contrib.slim


def resnet(inputs, n, num_classes=None, is_training=True, scope='resnet'):
    '''Resnet model'''

    with tf.variable_scope(scope) as sc:
        end_points_collections = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            output_collections=[end_points_collections]), \
                slim.arg_scope([slim.batch_norm], is_training=is_training):


def residual_block(inputs, out_channels, first_block=False):
    '''Residual block'''
    in_channels = inputs.get_shape().as_list()[-1]
