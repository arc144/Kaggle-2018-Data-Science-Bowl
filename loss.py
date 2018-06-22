import tensorflow as tf


class categorical_cross_entropy():
    '''Categorical cross entropy loss'''

    def __init__(self, use_weights=False, onehot_convert=True):
        self.use_weights = use_weights
        self.onehot_convert = onehot_convert

    def __call__(self, logits_tf, y_tf, w_tf=None):
        if self.onehot_convert:
            onehot_labels = tf.one_hot(tf.cast(tf.squeeze(y_tf), tf.uint8), 2)
        else:
            onehot_labels = y_tf

        if self.use_weights:
            weights = self.calc_class_weights(onehot_labels)
            if w_tf is not None:
                weights = tf.squeeze(w_tf) + weights
        else:
            weights = 1.0
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits_tf,
            weights=weights)
        loss = tf.reduce_mean(loss)
        return loss

    def calc_class_weights(self, onehot_labels):
        '''Compute the unbalacing factor class weights'''
        total = self.input_shape[0] * self.input_shape[1]
        values = tf.reduce_sum(onehot_labels,
                               axis=[1, 2],
                               keepdims=True) / total
        cls_weights = tf.reduce_sum(onehot_labels * values,
                                    axis=-1,
                                    name='cls_weights')
        return cls_weights


class soft_dice():
    '''Soft dice (IoU) loss
        exclude_background is used to avoid computing dice loss over
        background channel'''

    def __init__(self, exclude_background=True, is_onehot=False):
        self.exclude_background = exclude_background
        self.is_onehot = is_onehot

    def __call__(self, logits_tf, y_tf):
        smooth = 1
        if self.exclude_background:
            logits_tf = logits_tf[:, :, :, 1:]
            if self.is_onehot:
                y_tf = y_tf[:, :, :, 1:]
        prob = tf.nn.softmax(logits_tf)

        intersection = tf.reduce_sum(prob * y_tf)
        union = tf.reduce_sum(prob + y_tf)
        dice_coeff = (2. * intersection + smooth) / \
            (union + smooth)

        loss = 1 - tf.reduce_mean(dice_coeff)
        return loss
