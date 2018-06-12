    def build_UNet_graph(self, res=False):
        """ Create the UNet graph in TensorFlow. """
        # 1. unit
        with tf.name_scope('1.unit'):
            W1_1 = self.weight_variable(
                [3, 3, self.input_shape[2], 16], 'W1_1')
            b1_1 = self.bias_variable([16], 'b1_1')
            Z1 = self.conv2d(self.x_data_tf, W1_1, 'Z1') + b1_1
            A1 = self.activation(self.batch_norm_layer(Z1))  # (.,128,128,16)
            A1_drop = self.dropout_layer(A1)
            W1_2 = self.weight_variable([3, 3, 16, 16], 'W1_2')
            b1_2 = self.bias_variable([16], 'b1_2')
            Z2 = self.conv2d(A1_drop, W1_2, 'Z2') + b1_2
            A2 = self.activation(self.batch_norm_layer(Z2))  # (.,128,128,16)
            if res:
                WS_1 = self.weight_variable([1, 1, 3, 16], 'WS_1')
                A2 = tf.add(A2, self.conv2d(self.x_data_tf, WS_1))
            P1 = self.max_pool_2x2(A2, 'P1')  # (.,64,64,16)
        # 2. unit
        with tf.name_scope('2.unit'):
            W2_1 = self.weight_variable([3, 3, 16, 32], "W2_1")
            b2_1 = self.bias_variable([32], 'b2_1')
            Z3 = self.conv2d(P1, W2_1) + b2_1
            A3 = self.activation(self.batch_norm_layer(Z3))  # (.,64,64,32)
            A3_drop = self.dropout_layer(A3)
            W2_2 = self.weight_variable([3, 3, 32, 32], "W2_2")
            b2_2 = self.bias_variable([32], 'b2_2')
            Z4 = self.conv2d(A3_drop, W2_2) + b2_2
            A4 = self.activation(self.batch_norm_layer(Z4))  # (.,64,64,32)
            if res:
                WS_2 = self.weight_variable([1, 1, 16, 32], 'WS_2')
                A4 = tf.add(A4, self.conv2d(P1, WS_2))
            P2 = self.max_pool_2x2(A4)  # (.,32,32,32)
        # 3. unit
        with tf.name_scope('3.unit'):
            W3_1 = self.weight_variable([3, 3, 32, 64], "W3_1")
            b3_1 = self.bias_variable([64], 'b3_1')
            Z5 = self.conv2d(P2, W3_1) + b3_1
            A5 = self.activation(self.batch_norm_layer(Z5))  # (.,32,32,64)
            A5_drop = self.dropout_layer(A5)
            W3_2 = self.weight_variable([3, 3, 64, 64], "W3_2")
            b3_2 = self.bias_variable([64], 'b3_2')
            Z6 = self.conv2d(A5_drop, W3_2) + b3_2
            A6 = self.activation(self.batch_norm_layer(Z6))  # (.,32,32,64)
            if res:
                WS_3 = self.weight_variable([1, 1, 32, 64], 'WS_3')
                A6 = tf.add(A6, self.conv2d(P2, WS_3))
            P3 = self.max_pool_2x2(A6)  # (.,16,16,64)
        # 4. unit
        with tf.name_scope('4.unit'):
            W4_1 = self.weight_variable([3, 3, 64, 128], "W4_1")
            b4_1 = self.bias_variable([128], 'b4_1')
            Z7 = self.conv2d(P3, W4_1) + b4_1
            A7 = self.activation(self.batch_norm_layer(Z7))  # (.,16,16,128)
            A7_drop = self.dropout_layer(A7)
            W4_2 = self.weight_variable([3, 3, 128, 128], "W4_2")
            b4_2 = self.bias_variable([128], 'b4_2')
            Z8 = self.conv2d(A7_drop, W4_2) + b4_2
            A8 = self.activation(self.batch_norm_layer(Z8))  # (.,16,16,128)
            if res:
                WS_4 = self.weight_variable([1, 1, 64, 128], 'WS_4')
                A8 = tf.add(A8, self.conv2d(P3, WS_4))
            P4 = self.max_pool_2x2(A8)  # (.,8,8,128)
        # 5. unit
        with tf.name_scope('5.unit'):
            W5_1 = self.weight_variable([3, 3, 128, 256], "W5_1")
            b5_1 = self.bias_variable([256], 'b5_1')
            Z9 = self.conv2d(P4, W5_1) + b5_1
            A9 = self.activation(self.batch_norm_layer(Z9))  # (.,8,8,256)
            A9_drop = self.dropout_layer(A9)
            W5_2 = self.weight_variable([3, 3, 256, 256], "W5_2")
            b5_2 = self.bias_variable([256], 'b5_2')
            Z10 = self.conv2d(A9_drop, W5_2) + b5_2
            A10 = self.activation(self.batch_norm_layer(Z10))  # (.,8,8,256)
            if res:
                WS_5 = self.weight_variable([1, 1, 128, 256], 'WS_5')
                A10 = tf.add(A10, self.conv2d(P4, WS_5))
            P5 = self.max_pool_2x2(A10)  # (.,4,4,128)
        # 6. unit
        with tf.name_scope('6.unit'):
            W5e_1 = self.weight_variable([3, 3, 256, 512], "W5e_1")
            b5e_1 = self.bias_variable([512], 'b5e_1')
            Z9e = self.conv2d(P5, W5e_1) + b5e_1
            A9e = self.activation(self.batch_norm_layer(Z9e))  # (.,4,4,512)
            A9e_drop = self.dropout_layer(A9e)
            W5e_2 = self.weight_variable([3, 3, 512, 512], "W5e_2")
            b5e_2 = self.bias_variable([512], 'b5e_2')
            Z10e = self.conv2d(A9e_drop, W5e_2) + b5e_2
            A10e = self.activation(self.batch_norm_layer(Z10e))  # (.,4,4,512)
            if res:
                WSe_5 = self.weight_variable([1, 1, 256, 512], 'WSe_5')
                A10e = tf.add(A10e, self.conv2d(P5, WSe_5))

        with tf.name_scope('7.unit'):
            W6e_1 = self.weight_variable([3, 3, 512, 256], "W6e_1")
            b6e_1 = self.bias_variable([256], 'b6e_1')
            U1e = self.conv2d_transpose(A10e, 256)  # (.,8,8,256)
            U1e = tf.concat([U1e, A10], 3)  # (.,8,8,512)
            Z11e = self.conv2d(U1e, W6e_1) + b6e_1
            A11e = self.activation(self.batch_norm_layer(Z11e))  # (.,8,8,256)
            A11e_drop = self.dropout_layer(A11e)
            W6e_2 = self.weight_variable([3, 3, 256, 256], "W6e_2")
            b6e_2 = self.bias_variable([256], 'b6e_2')
            Z12e = self.conv2d(A11e_drop, W6e_2) + b6e_2
            A12e = self.activation(self.batch_norm_layer(Z12e))  # (.,8,8,256)
            if res:
                WSe_6 = self.weight_variable([1, 1, 512, 256], 'WSe_6')
                A12e = tf.add(A12e, self.conv2d(U1e, WSe_6))

        with tf.name_scope('8.unit'):
            W6_1 = self.weight_variable([3, 3, 256, 128], "W6_1")
            b6_1 = self.bias_variable([128], 'b6_1')
            U1 = self.conv2d_transpose(A10, 128)  # (.,16,16,128)
            U1 = tf.concat([U1, A8], 3)  # (.,16,16,256)
            Z11 = self.conv2d(U1, W6_1) + b6_1
            A11 = self.activation(self.batch_norm_layer(Z11))  # (.,16,16,128)
            A11_drop = self.dropout_layer(A11)
            W6_2 = self.weight_variable([3, 3, 128, 128], "W6_2")
            b6_2 = self.bias_variable([128], 'b6_2')
            Z12 = self.conv2d(A11_drop, W6_2) + b6_2
            A12 = self.activation(self.batch_norm_layer(Z12))  # (.,16,16,128)
            if res:
                WS_6 = self.weight_variable([1, 1, 256, 128], 'WS_6')
                A12 = tf.add(A12, self.conv2d(U1, WS_6))
        # 9. unit
        with tf.name_scope('9.unit'):
            W7_1 = self.weight_variable([3, 3, 128, 64], "W7_1")
            b7_1 = self.bias_variable([64], 'b7_1')
            U2 = self.conv2d_transpose(A12, 64)  # (.,32,32,64)
            U2 = tf.concat([U2, A6], 3)  # (.,32,32,128)
            Z13 = self.conv2d(U2, W7_1) + b7_1
            A13 = self.activation(self.batch_norm_layer(Z13))  # (.,32,32,64)
            A13_drop = self.dropout_layer(A13)
            W7_2 = self.weight_variable([3, 3, 64, 64], "W7_2")
            b7_2 = self.bias_variable([64], 'b7_2')
            Z14 = self.conv2d(A13_drop, W7_2) + b7_2
            A14 = self.activation(self.batch_norm_layer(Z14))  # (.,32,32,64)
            if res:
                WS_7 = self.weight_variable([1, 1, 128, 64], 'WS_7')
                A14 = tf.add(A14, self.conv2d(U2, WS_7))
        # 10. unit
        with tf.name_scope('10.unit'):
            W8_1 = self.weight_variable([3, 3, 64, 32], "W8_1")
            b8_1 = self.bias_variable([32], 'b8_1')
            U3 = self.conv2d_transpose(A14, 32)  # (.,64,64,32)
            U3 = tf.concat([U3, A4], 3)  # (.,64,64,64)
            Z15 = self.conv2d(U3, W8_1) + b8_1
            A15 = self.activation(self.batch_norm_layer(Z15))  # (.,64,64,32)
            A15_drop = self.dropout_layer(A15)
            W8_2 = self.weight_variable([3, 3, 32, 32], "W8_2")
            b8_2 = self.bias_variable([32], 'b8_2')
            Z16 = self.conv2d(A15_drop, W8_2) + b8_2
            A16 = self.activation(self.batch_norm_layer(Z16))  # (.,64,64,32)
            if res:
                WS_8 = self.weight_variable([1, 1, 64, 32], 'WS_8')
                A16 = tf.add(A16, self.conv2d(U3, WS_8))
        # 11. unit
        with tf.name_scope('11.unit'):
            W9_1 = self.weight_variable([3, 3, 32, 16], "W9_1")
            b9_1 = self.bias_variable([16], 'b9_1')
            U4 = self.conv2d_transpose(A16, 16)  # (.,128,128,16)
            U4 = tf.concat([U4, A2], 3)  # (.,128,128,32)
            Z17 = self.conv2d(U4, W9_1) + b9_1
            A17 = self.activation(self.batch_norm_layer(Z17))  # (.,128,128,16)
            A17_drop = self.dropout_layer(A17)
            W9_2 = self.weight_variable([3, 3, 16, 16], "W9_2")
            b9_2 = self.bias_variable([16], 'b9_2')
            Z18 = self.conv2d(A17_drop, W9_2) + b9_2
            A18 = self.activation(self.batch_norm_layer(Z18))  # (.,128,128,16)
            if res:
                WS_9 = self.weight_variable([1, 1, 32, 16], 'WS_9')
                A18 = tf.add(A18, self.conv2d(U4, WS_9))
        # 12. unit: output layer
        with tf.name_scope('12.unit'):
            W10 = self.weight_variable([1, 1, 16, 1], "W10")
            b10 = self.bias_variable([1], 'b10')
            Z19 = self.conv2d(A18, W10, pad=0) + b10
            A19 = tf.nn.sigmoid(self.batch_norm_layer(Z19))  # (.,128,128,1)

        self.z_pred_tf = tf.identity(Z19, name='z_pred_tf')  # (.,128,128,1)
        self.y_pred_tf = tf.identity(A19, name='y_pred_tf')  # (.,128,128,1)

        print('Build UNet Graph: 10 layers, {} trainable weights'.format(
            self.num_of_weights([W1_1, b1_1, W1_2, b1_2, W2_1, b2_1, W2_2, b2_2,
                                 W3_1, b3_1, W3_2, b3_2, W4_1, b4_1, W4_2, b4_2,
                                 W5_1, b5_1, W5_2, b5_2, W6_1, b6_1, W6_2, b6_2,
                                 W7_1, b7_1, W7_2, b7_2, W8_1, b8_1, W8_2, b8_2,
                                 W9_1, b9_1, W9_2, b9_2, W10, b10])))

    def loss_tensor(self):
        """Loss tensor."""
        if LOSS == 0:
            # Dice loss based on Jaccard dice score coefficent.
            axis = np.arange(1, len(self.output_shape) + 1)
            offset = 1e-5
            corr = tf.reduce_sum(self.y_data_tf * self.logits_tf, axis=axis)
            l2_pred = tf.reduce_sum(tf.square(self.logits_tf), axis=axis)
            l2_true = tf.reduce_sum(tf.square(self.y_data_tf), axis=axis)
            dice_coeff = (2. * corr + 1e-5) / (l2_true + l2_pred + 1e-5)
            # Second version: 2-class variant of dice loss
            #corr_inv = tf.reduce_sum((1.-self.y_data_tf) * (1.-self.logits_tf), axis=axis)
            #l2_pred_inv = tf.reduce_sum(tf.square(1.-self.logits_tf), axis=axis)
            #l2_true_inv = tf.reduce_sum(tf.square(1.-self.y_data_tf), axis=axis)
            # dice_coeff = ((corr + offset) / (l2_true + l2_pred + offset) +
            #             (corr_inv + offset) / (l2_pred_inv + l2_true_inv + offset))
            loss = tf.subtract(1., tf.reduce_mean(dice_coeff))
        if LOSS == 1:
            # Sigmoid cross entropy.
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.y_data_tf, logits=self.z_pred_tf))

        if LOSS == 2:
            # Sparse softmax
            logits = tf.reshape(self.z_pred_tf, [-1, 1])
            labels = tf.cast(tf.reshape(self.y_data_tf, [-1]), tf.int32)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
        if LOSS == 3:
            # Weighted sigmoid
            logits = tf.reshape(self.z_pred_tf, [-1, 1])
            labels = tf.reshape(self.y_data_tf, [-1, 1])
            weights = tf.reshape(self.w_data_tf, [-1, 1])
            loss = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                                logits=logits,
                                                weights=weights))
        return loss

    def next_mini_batch(self):
        """Get the next mini batch."""
        start = self.index_in_epoch
        self.index_in_epoch += self.mb_size
        self.epoch += self.mb_size / len(self.x_train)

        # At the start of the epoch.
        if start == 0:
            np.random.shuffle(self.perm_array)  # Shuffle permutation array.

        # In case the current index is larger than one epoch.
        if self.index_in_epoch > len(self.x_train):
            self.index_in_epoch = 0
            self.epoch -= self.mb_size / len(self.x_train)
            return self.next_mini_batch()  # Recursive use of function.

        end = self.index_in_epoch

        # Original data.
        x_tr = self.x_train[self.perm_array[start:end]]
        y_tr = self.y_train[self.perm_array[start:end]]
        y_wg = self.w_train[self.perm_array[start:end]]

        # Use augmented data.
        if self.train_on_augmented_data:
            x_tr, y_tr, y_wg = uf.generate_images_and_masks(x_tr, y_tr, y_wg)
            y_tr = uf.trsf_proba_to_binary(y_tr)

        return x_tr, y_tr, y_wg


def batch_norm_layer(self, x, name=None):
    """Batch normalization layer."""
    if self.use_bn:
        layer = tf.layers.batch_normalization(x, training=self.training_tf,
                                              momentum=0.9, name=name)
    else:
        layer = x
    return layer


def dropout_layer(self, x, name=None):
    """Dropout layer."""
    if self.use_drop:
        layer = tf.layers.dropout(x, self.dropout_proba,
                                  training=self.training_tf,
                                  name=name)
    else:
        layer = x
    return layer


def weight_variable(self, shape, name=None):
    """ Weight initialization """
    # initializer = tf.truncated_normal(shape, stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    # initializer = tf.contrib.layers.variance_scaling_initializer()
    return tf.get_variable(name, shape=shape, initializer=initializer)


def bias_variable(self, shape, name=None):
    """Bias initialization."""
    # initializer = tf.constant(0.1, shape=shape)
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    # initializer = tf.contrib.layers.variance_scaling_initializer()
    return tf.get_variable(name, shape=shape, initializer=initializer)


def conv2d(self, x, W, name=None, pad=1):
    """ 2D convolution. """
    padding = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    padded_x = tf.pad(x, padding, 'reflect')
    return tf.nn.conv2d(padded_x, W, strides=[1, 1, 1, 1],
                        padding='VALID', name=name)


def max_pool_2x2(self, x, name=None):
    """ Max Pooling 2x2. """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)


def conv2d_transpose(self, x, filters, name=None):
    """ Transposed 2d convolution. """
    return tf.layers.conv2d_transpose(x, filters=filters, kernel_size=2,
                                      strides=2, padding='SAME')


def activation(self, x, act_fun, name=None):
    """ Activation function. """
    if act_fun == 'elu':
        a = tf.nn.elu(x, name=name)

    elif act_fun == 'leaky_relu':
        a = self.leaky_relu(x, name=name)

    elif act_fun == 'relu':
        a = tf.nn.relu(x, name=name)

    elif act_fun == 'selu':
        a = tf.nn.selu(x, name=name)

    return a
