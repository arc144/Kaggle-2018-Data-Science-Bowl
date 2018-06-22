import tensorflow as tf
import numpy as np
import datetime
import os
import utils
import score
import unets
import tempfile


class ConvNetwork_ABC():
    """ Implements a neural network.
        Methods are implemented to train the model,
        to save/load the complete session and to
        attach summaries for visualization with TensorBoard.
    """

    def __init__(self, nn_name='tmp', log_step=0.2,
                 keep_prob=0.8, mb_size=5,
                 input_shape=None, output_shape=None,
                 learn_rate_0=0.001, learn_rate_alpha=0.25, learn_rate_step=3,
                 activation_fun='selu', padding='REFLECT',
                 use_bn=False, use_drop=False, use_weights=False,
                 dir_dict=None):
        """Instance constructor."""

        # Tunable hyperparameters for training.
        self.mb_size = mb_size       # Mini batch size
        self.keep_prob = keep_prob   # Keeping probability with dropout regularization
        self.learn_rate_step = learn_rate_step    # Step size in terms of epochs
        self.activation_fun = activation_fun
        self.padding = padding
        # Reduction of learn rate for each step
        self.learn_rate_alpha = learn_rate_alpha
        self.learn_rate_0 = learn_rate_0    # Starting learning rate
        self.dropout_proba = 1 - keep_prob
        # Batch norm and dropout
        self.use_bn = use_bn
        self.use_drop = use_drop
        self.use_weights = use_weights

        # Set helper variables.
        assert [input_shape, output_shape] is not None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nn_name = nn_name                # Name of neural network
        self.params = {}                      # For storing parameters
        self.learn_rate_pos = 0
        self.learn_rate = self.learn_rate_0
        self.index_in_epoch = 0
        self.epoch = 0.
        self.log_step = log_step              # Log results in terms of epochs
        self.n_log_step = 0                   # Count number of mini batches
        self.train_on_augmented_data = True  # True = use augmented data
        self.use_tb_summary = False           # True = use TensorBoard summaries
        self.use_tf_saver = False             # True = save the session
        self.dir_dict = dir_dict

        # Parameters that should be stored.
        self.params['train_loss'] = []
        self.params['valid_loss'] = []
        self.params['train_score'] = []
        self.params['valid_score'] = []
        self.params['train_iou'] = []
        self.params['valid_iou'] = []

    def get_learn_rate(self):
        """Compute the current learning rate."""
        if False:
            # Fixed learnrate
            learn_rate = self.learn_rate_0
        else:
            # Decreasing learnrate each step by factor 1-alpha
            learn_rate = self.learn_rate_0 * \
                (1. - self.learn_rate_alpha)**self.learn_rate_pos
        return learn_rate

    def leaky_relu(self, z, name=None):
        """Leaky ReLU."""
        return tf.maximum(0.01 * z, z, name=name)

    def optimizer_tensor(self):
        """Optimization tensor."""
        # Adam Optimizer (adaptive moment estimation).
        optimizer = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
            self.loss_tf, var_list=self.train_list, name='train_step_tf')
        return optimizer

    def num_of_weights(self, tensors):
        """Compute the number of weights."""
        sum_ = 0
        for i in range(len(tensors)):
            m = 1
            for j in range(len(tensors[i].shape)):
                m *= int(tensors[i].shape[j])
            sum_ += m
        return sum_

    def build_graph(self):
        """ Build the complete graph in TensorFlow. """
        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Input tensor.
            shape = [None]
            shape.extend(self.input_shape)
            self.x_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='x_tf')  # (.,128,128,3)

            # Generic tensors.
            self.keep_prob_tf = tf.placeholder_with_default(
                1.0,
                shape=(),
                name='keep_prob_tf')
            self.learn_rate_tf = tf.placeholder(dtype=tf.float32,
                                                name="learn_rate_tf")
            self.training_tf = tf.placeholder_with_default(False,
                                                           shape=(),
                                                           name='training_tf')
            # Build U-Net graph.
            self.logits_tf = tf.identity(self.create_graph(), name='logits_tf')

            # Target tensor.
            shape = [None]
            shape.extend(self.output_shape)
            self.y_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='y_tf')  # (.,128,128,1)

            # Weights tensor.
            shape = [None, self.input_shape[0], self.input_shape[1], 1]
            self.w_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='w_tf')  # (.,128,128,1)

            # Loss tensor
            self.loss_tf = tf.identity(self.loss_tensor(), name='loss_tf')

            # Optimisation tensor.
            self.train_step_tf = self.optimizer_tensor()

            # Extra operations required for batch normalization.
            self.extra_update_ops_tf = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)

    def create_graph(self):
        '''Reimplement this class with your graph and
        return the unscaled logits'''
        pass

    def loss_tensor(self):
        '''Reimplement this function and return the loss'''
        pass

    def next_mini_batch(self):
        '''Reimplement this function and return the x,y,w(optional) batches'''
        pass

    def get_score(self, logits, y):
        '''Reimplement this function and return the score'''
        pass

    def train_graph(self, sess,
                    x_train, y_train,
                    x_valid, y_valid,
                    w_train=None, w_valid=None,
                    n_epoch=1, train_on_augmented_data=False,
                    lr=None):
        """ Train the graph of the corresponding neural network. """
        # if lr, overwrite self.learn_rate and reset self.learn_rate_pos
        if lr is not None:
            self.learn_rate_0 = lr
            self.learn_rate_pos = 0
        # Set training and validation sets.
        self.x_train = x_train
        self.y_train = y_train

        self.x_valid = x_valid
        self.y_valid = y_valid

        self.w_train = w_train
        self.w_valid = w_valid

        # Parameters.
        self.perm_array = np.arange(len(self.x_train))
        self.train_on_augmented_data = train_on_augmented_data
        mb_per_epoch = self.x_train.shape[0] / self.mb_size

        # Start timer.
        start = datetime.datetime.now()
        print('Training the Neural Network')
        print('\tnn_name = {}, n_epoch = {}, mb_size = {}, learnrate = {:.7f}'.format(
            self.nn_name, n_epoch, self.mb_size, self.learn_rate))
        print('\tinput_shape = {}, output_shape = {}'.format(
            self.input_shape, self.output_shape))
        print('\tlearn_rate = {:.10f}, learn_rate_0 = {:.10f}, learn_rate_alpha = {}'.format(
            self.learn_rate, self.learn_rate_0, self.learn_rate_alpha))
        print('\tlearn_rate_step = {}, learn_rate_pos = {}, dropout_proba = {}'.format(
            self.learn_rate_step, self.learn_rate_pos, self.dropout_proba))
        print('\tx_train = {}, x_valid = {}'.format(
            x_train.shape, x_valid.shape))
        print('\ty_train = {}, y_valid = {}'.format(
            y_train.shape, y_valid.shape))
        print('Training started: {}'.format(datetime.datetime.now().strftime(
            '%d-%m-%Y %H:%M:%S')))

        # Looping over mini batches.
        for i in range(int(n_epoch * mb_per_epoch) + 1):

            # Adapt the learning rate.
            if not self.learn_rate_pos == int(self.epoch // self.learn_rate_step):
                self.learn_rate_pos += 1
                self.learn_rate = self.get_learn_rate()
                print('Update learning rate to {:.10f}. Running time: {}'.format(
                    self.learn_rate, datetime.datetime.now() - start))

            # Train the graph.
            x_batch, y_batch, w_batch = self.next_mini_batch()  # next mini batch
            sess.run([self.train_step_tf, self.extra_update_ops_tf],
                     feed_dict={self.x_tf: x_batch,
                                self.y_tf: y_batch,
                                self.w_tf: w_batch,
                                self.keep_prob_tf: self.keep_prob,
                                self.learn_rate_tf: self.learn_rate,
                                tf.keras.backend.learning_phase(): 1,  # Required if using tf.keras
                                self.training_tf: True})

            # Store losses and scores.
            if i % int(self.log_step * mb_per_epoch) == 0:

                self.n_log_step += 1  # Current number of log steps.

                trn_dct = dict(x=self.x_train, y=self.y_train, w=self.w_train)
                val_dct = dict(x=self.x_valid, y=self.y_valid, w=self.w_valid)
                for dct in [trn_dct, val_dct]:
                    # Random ids for eval (same size as val)
                    ids = np.arange(len(dct['x']))
                    np.random.shuffle(ids)
                    ids = ids[:len(val_dct['x'])]  # len(x_batch)
                    x = dct['x'][ids]
                    y = dct['y'][ids]
                    w = dct['w'][ids]

                    dct['feed_dict'] = {self.x_tf: x,
                                        self.y_tf: y,
                                        self.w_tf: w,
                                        self.keep_prob_tf: 1.0}

                    # Evaluate current loss and score
                    dct['loss'] = sess.run(self.loss_tf,
                                           feed_dict=dct['feed_dict'])
                    pred = self.get_prediction(sess, x)
                    dct['score'] = np.mean(self.get_score(pred, y))

                print(('{:.2f} epoch: train/valid loss = {:.4f}/{:.4f} ' +
                       'train/valid score = {:.4f}/{:.4f}').format(
                    self.epoch, trn_dct['loss'], val_dct['loss'],
                      trn_dct['score'], val_dct['score']))

                # Store losses and scores.
                self.params['train_loss'].extend([trn_dct['loss']])
                self.params['valid_loss'].extend([val_dct['loss']])
                self.params['train_score'].extend([trn_dct['score']])
                self.params['valid_score'].extend([val_dct['score']])

                # Save summaries for TensorBoard.
                if self.use_tb_summary:
                    train_summary = sess.run(
                        self.merged, feed_dict=trn_dct['feed_dict'])
                    valid_summary = sess.run(
                        self.merged, feed_dict=val_dct['feed_dict'])
                    self.train_writer.add_summary(
                        train_summary, self.n_log_step)
                    self.valid_writer.add_summary(
                        valid_summary, self.n_log_step)

        # Store parameters.
        self.params['learn_rate'] = self.learn_rate
        self.params['learn_rate_step'] = self.learn_rate_step
        self.params['learn_rate_pos'] = self.learn_rate_pos
        self.params['learn_rate_alpha'] = self.learn_rate_alpha
        self.params['learn_rate_0'] = self.learn_rate_0
        self.params['keep_prob'] = self.keep_prob
        self.params['epoch'] = self.epoch
        self.params['n_log_step'] = self.n_log_step
        self.params['log_step'] = self.log_step
        self.params['input_shape'] = self.input_shape
        self.params['output_shape'] = self.output_shape
        self.params['mb_size'] = self.mb_size
        self.params['dropout_proba'] = self.dropout_proba

        print('Training ended. Running time: {}'.format(
            datetime.datetime.now() - start))

    def summary_variable(self, var, var_name):
        """ Attach summaries to a tensor for TensorBoard visualization. """
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def attach_summary(self, sess):
        """ Attach TensorBoard summaries to certain tensors. """
        self.use_tb_summary = True

        # Create summary tensors for TensorBoard.
        tf.summary.scalar('loss_tf', self.loss_tf)

        # Merge all summaries.
        self.merged = tf.summary.merge_all()

        # Initialize summary writer.
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(
            os.getcwd(), self.dir_dict['logs'],
            (self.nn_name + '_' + timestamp))
        self.train_writer = tf.summary.FileWriter(
            os.path.join(filepath, 'train'), sess.graph)
        self.valid_writer = tf.summary.FileWriter(
            os.path.join(filepath, 'valid'), sess.graph)

    def attach_saver(self):
        """ Initialize TensorFlow saver. """
        with self.graph.as_default():
            self.use_tf_saver = True
            self.saver_tf = tf.train.Saver()

    def save_model(self, sess):
        """ Save parameters, tensors and summaries. """
        if not os.path.isdir(os.path.join(os.getcwd(),
                                          self.dir_dict['saves'])):
            os.mkdir(self.dir_dict['saves'])
        filepath = os.path.join(
            os.getcwd(), self.dir_dict['saves'], self.nn_name + '_params.npy')
        np.save(filepath, self.params)  # save parameters of the network

        # TensorFlow saver
        if self.use_tf_saver:
            filepath = os.path.join(os.getcwd(),  self.nn_name)
            self.saver_tf.save(sess, filepath)

        # TensorBoard summaries
        if self.use_tb_summary:
            self.train_writer.close()
            self.valid_writer.close()

    def load_session_from_file(self, filename,
                               update_cost=False, renew_LR=False):
        """ Load session from a file, restore the graph, and load the tensors."""
        tf.reset_default_graph()
        filepath = os.path.join(os.getcwd(), filename + '.meta')
        saver = tf.train.import_meta_graph(filepath)
        sess = tf.Session()  # default session
        saver.restore(sess, filename)  # restore session
        self.graph = tf.get_default_graph()  # save default graph
        self.load_parameters(filename)  # load parameters
        if renew_LR is not False:
            self.learn_rate_step = 0
            self.learn_rate_0 = renew_LR
            self.epoch = 0
        # define relevant tensors as variables
        self.load_tensors(self.graph, update_cost)
        return sess

    def load_parameters(self, filename):
        '''Load helper and tunable parameters.'''
        filepath = os.path.join(
            os.getcwd(), self.dir_dict['saves'], filename + '_params.npy')
        self.params = np.load(filepath).item()  # load parameters of network

        self.nn_name = filename
        self.learn_rate = self.params['learn_rate']
        self.learn_rate_0 = self.params['learn_rate_0']
        self.learn_rate_step = self.params['learn_rate_step']
        self.learn_rate_alpha = self.params['learn_rate_alpha']
        self.learn_rate_pos = self.params['learn_rate_pos']
        self.keep_prob = self.params['keep_prob']
        self.epoch = self.params['epoch']
        self.n_log_step = self.params['n_log_step']
        self.log_step = self.params['log_step']
        self.input_shape = self.params['input_shape']
        self.output_shape = self.params['output_shape']
        self.mb_size = self.params['mb_size']
        self.dropout_proba = self.params['dropout_proba']

        print('Parameters of the loaded neural network')
        print('\tnn_name = {}, epoch = {:.2f}, mb_size = {}'.format(
            self.nn_name, self.epoch, self.mb_size))
        print('\tinput_shape = {}, output_shape = {}'.format(
            self.input_shape, self.output_shape))
        print('\tlearn_rate = {:.10f}, learn_rate_0 = {:.10f}, dropout_proba = {}'.format(
            self.learn_rate, self.learn_rate_0, self.dropout_proba))
        print('\tlearn_rate_step = {}, learn_rate_pos = {}, learn_rate_alpha = {}'.format(
            self.learn_rate_step, self.learn_rate_pos, self.learn_rate_alpha))

    def load_tensors(self, graph, update_cost=False):
        """ Load tensors from a graph. """
        # Input tensors
        self.x_tf = graph.get_tensor_by_name("x_tf:0")
        self.y_tf = graph.get_tensor_by_name("y_tf:0")
        self.w_tf = graph.get_tensor_by_name("w_tf:0")

        # Tensors for training and prediction.
        self.learn_rate_tf = graph.get_tensor_by_name("learn_rate_tf:0")
        self.keep_prob_tf = graph.get_tensor_by_name("keep_prob_tf:0")
        self.train_step_tf = graph.get_operation_by_name('train_step_tf')
        self.logits_tf = graph.get_tensor_by_name("logits_tf:0")
        if not update_cost:
            self.loss_tf = graph.get_tensor_by_name('loss_tf:0')
        else:
            self.loss_tf = self.loss_tensor()
        self.training_tf = graph.get_tensor_by_name("training_tf:0")
        self.extra_update_ops_tf = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def get_prediction(self, sess, x_data, keep_prob=1.0):
        """ Prediction of the neural network graph. """
        pred = sess.run(self.logits_tf,
                        feed_dict={self.x_tf: x_data,
                                   self.keep_prob_tf: keep_prob})
        return pred

    def get_loss(self, sess, x_data, y_data, keep_prob=1.0):
        """ Compute the loss. """
        return sess.run(self.loss_tf, feed_dict={self.x_tf: x_data,
                                                 self.y_tf: y_data,
                                                 self.keep_prob_tf: keep_prob})


class U_Net(ConvNetwork_ABC):

    def __init__(self, *args, loss='ce', net_type='vanilla', ** kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.net_type = net_type
        self.buffer_x = []
        self.buffer_y = []

    def build_graph(self):
        """ Build the complete graph in TensorFlow. """
        tf.keras.backend.clear_session()
        tf.reset_default_graph()
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Input tensor.
            shape = [None, None, None, self.input_shape[-1]]
            # shape.extend(self.input_shape)
            self.x_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='x_tf')  # (.,128,128,3)

            # Generic tensors.
            self.keep_prob_tf = tf.placeholder_with_default(
                1.0,
                shape=(),
                name='keep_prob_tf')
            self.learn_rate_tf = tf.placeholder(dtype=tf.float32,
                                                name="learn_rate_tf")
            self.training_tf = tf.placeholder_with_default(False,
                                                           shape=(),
                                                           name='training_tf')
            # Build U-Net graph.
            self.logits_tf = tf.identity(self.create_graph(), name='logits_tf')

            # Target tensor.
            shape = [None, None, None, self.output_shape[-1]]
            # shape.extend(self.output_shape)
            self.y_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='y_tf')  # (.,128,128,1)

            # Weights tensor.
            shape = [None, self.input_shape[0], self.input_shape[1], 1]
            self.w_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='w_tf')  # (.,128,128,1)

            # Loss tensor
            self.loss_tf = tf.identity(self.loss_tensor(), name='loss_tf')

            # Optimisation tensor.
            self.train_step_all_tf, self.train_step_top_tf = self.optimizer_tensor()

            # Extra operations required for batch normalization.
            self.extra_update_ops_tf = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)

    def create_graph(self):
        '''Reimplement this class with your graph and
        return the unscaled logits'''
        if self.net_type == 'vanilla':
            logits, self.end_points = \
                unets.vanilla(self.x_tf,
                              activation_fn=self.activation_fun,
                              padding=self.padding)
        else:
            if self.net_type == 'Xception_vanilla':
                logits, self.encoder, self.end_points = \
                    unets.Xception_vanilla(self.x_tf,
                                           activation_fn=self.activation_fun,
                                           padding=self.padding)

            elif self.net_type == 'SE_Xception_vanillaSE':
                logits, self.encoder, self.end_points = \
                    unets.SE_Xception_vanillaSE(self.x_tf,
                                                activation_fn=self.activation_fun,
                                                padding=self.padding)

            elif self.net_type == 'Xception_InceptionSE':
                logits, self.encoder, self.end_points = \
                    unets.Xception_InceptionSE(self.x_tf,
                                               activation_fn=self.activation_fun,
                                               padding=self.padding)

            elif self.net_type == 'InceptionResNetV2_vanilla':
                logits, self.encoder, self.end_points = \
                    unets.InceptionResNetV2(self.x_tf,
                                            activation_fn=self.activation_fun,
                                            padding=self.padding)

            self.weights = [w.trainable_weights for w in self.encoder.layers
                            if len(w.trainable_weights) > 0]
            self.weights = [x for y in self.weights for x in y]
            self.non_initialized_weights = [
                w for w in tf.trainable_variables() if w not in self.weights]
            sess = tf.keras.backend.get_session()
            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(
                    self.weights).save(sess, f.name)

            self.model_weights_tensors = set(self.weights)

        return logits

    def loss_tensor(self):
        '''Reimplement this function and return the loss'''
        # Softmax
        # logits = tf.reshape(self.logits_tf, [-1, 1])
        # labels = tf.cast(tf.reshape(self.y_tf, [-1]), tf.int32)
        onehot_labels = tf.one_hot(tf.cast(tf.squeeze(self.y_tf), tf.uint8), 2)
        if self.loss == 'ce':
            if self.use_weights:
                cls_weights = self.calc_class_weights(onehot_labels)
                weights = tf.squeeze(self.w_tf) + cls_weights
            else:
                weights = 1.0
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=self.logits_tf,
                weights=weights)
            loss = tf.reduce_mean(loss)

        elif self.loss == 'dice':
            # Dice loss based on dice score coefficent.
            axis = np.arange(1, len(self.output_shape) + 1)
            epilson = 1e-5
            corr = tf.reduce_sum(onehot_labels * self.logits_tf, axis=axis)
            l2_pred = tf.reduce_sum(tf.square(self.logits_tf), axis=axis)
            l2_true = tf.reduce_sum(tf.square(onehot_labels), axis=axis)
            dice_coeff = (2. * corr + epilson) / (l2_true + l2_pred + epilson)
            loss = tf.subtract(1., tf.reduce_mean(dice_coeff))

        elif self.loss == 'wdice':
            # Dice loss based on balanced dice score coefficent.
            axis = [1, 2]
            epilson = 1e-5
            total = self.output_shape[0] * self.output_shape[1]
            w = tf.reduce_sum(onehot_labels, axis=axis) / total
            I = tf.reduce_sum(onehot_labels * self.logits_tf, axis=axis)
            balanced_I = I * (1 - w + epilson)
            l2_pred = tf.reduce_sum(tf.square(self.logits_tf), axis=axis)
            l2_true = tf.reduce_sum(tf.square(onehot_labels), axis=axis)
            dice_coeff = (2. * balanced_I + epilson) / \
                (l2_true + l2_pred + epilson)
            loss = tf.subtract(1., tf.reduce_mean(dice_coeff))

        elif self.loss == 'ce+wdice':
            # Binary cross entropy
            if self.use_weights:
                cls_weights = self.calc_class_weights(onehot_labels)
                weights = tf.squeeze(self.w_tf) + cls_weights
            else:
                weights = 1.0
            loss1 = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=self.logits_tf,
                weights=weights)
            loss1 = tf.reduce_mean(loss1)

            # Soft dice loss
            axis = [1, 2]
            epilson = 1e-5
            total = self.output_shape[0] * self.output_shape[1]
            w = tf.reduce_sum(onehot_labels, axis=axis) / total
            I = tf.reduce_sum(onehot_labels * self.logits_tf, axis=axis)
            balanced_I = I * (1 - w + epilson)
            l2_pred = tf.reduce_sum(tf.square(self.logits_tf), axis=axis)
            l2_true = tf.reduce_sum(tf.square(onehot_labels), axis=axis)
            dice_coeff = (2. * balanced_I + epilson) / \
                (l2_true + l2_pred + epilson)
            loss2 = tf.subtract(1., tf.reduce_mean(dice_coeff))

            loss = loss1 + loss2

        elif self.loss == 'wdice+ce+entropy_penalty':
            epilson = 1e-5
            beta = 1.0
            # ce
            loss0 = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=self.logits_tf)
            loss0 = tf.reduce_mean(loss0)
            # entropy penalty
            prob = tf.nn.softmax(self.logits_tf) * onehot_labels
            entropy = - prob * tf.log(prob + epilson) / tf.log(10.)
            loss1 = tf.reduce_mean(beta * entropy)

            # Soft dice loss
            logits_tf = tf.expand_dims(self.logits_tf[:, :, :, 1], axis=-1)
            axis = [1, 2]
            total = self.output_shape[0] * self.output_shape[1]
            w = tf.reduce_sum(self.y_tf, axis=axis) / total
            I = tf.reduce_sum(self.y_tf * logits_tf, axis=axis)
            balanced_I = I * (1 - w + epilson)
            l2_pred = tf.reduce_sum(tf.square(logits_tf), axis=axis)
            l2_true = tf.reduce_sum(tf.square(self.y_tf), axis=axis)
            dice_coeff = (2. * balanced_I + epilson) / \
                (l2_true + l2_pred + epilson)
            loss2 = tf.subtract(1., tf.reduce_mean(dice_coeff))

            loss = loss0 + loss1 + loss2

        elif self.loss == 'focal':
            gamma = 2
            epilson = 1e-5

            ce = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_tf,
                                                         labels=self.y_tf)
            probt = tf.exp(-ce)
            loss = tf.pow((1 - probt), gamma) * ce
            loss = tf.reduce_mean(loss)

        return loss

    def optimizer_tensor(self):
        """Optimization tensor."""
        # Adam Optimizer (adaptive moment estimation).
        optimizer_all = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
            self.loss_tf, var_list=None, name='train_step_tf')

        if self.net_type == 'vanilla':
            optimizer_top = None
        else:
            optimizer_top = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
                self.loss_tf,
                var_list=self.non_initialized_weights,
                name='train_step_tf')
        return optimizer_all, optimizer_top

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

    def next_mini_batch(self, method, tgt_size):
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
            # Recursive use of function.
            return self.next_mini_batch(method, tgt_size)

        end = self.index_in_epoch

        # Original data.

        x_tr, y_tr = utils.load_images_masks(
            self.x_train[self.perm_array[start:end]],
            self.y_train[self.perm_array[start:end]],
            method=method, tgt_size=tgt_size)

        # Use augmented data.
        if self.train_on_augmented_data:
            x_tr, y_tr = utils.augment_images_masks(x_tr, y_tr)
            y_tr = utils.trsf_proba_to_binary(y_tr)

        return x_tr, y_tr, None

    def get_score(self, pred, y, from_paths=False, tgt_size=None, method=None):
        '''Reimplement this function and return the score'''
        # for i in range(len(pred)):
        #     pred[i] = pred[i] / pred[i].max()
        if from_paths:
            y = utils.load_masks(y, tgt_size=tgt_size, method=method)
        pred = utils.trsf_proba_to_binary(pred)
        return score.get_score(y, pred)

    def get_IoU(self, pred, y):
        for i in range(len(pred)):
            pred[i] = pred[i] / pred[i].max()
        pred = utils.trsf_proba_to_binary(pred)
        if len(pred.shape) == 3:
            y = np.squeeze(y)

        intersection = np.sum(pred * y)
        union = np.sum(np.maximum(pred, y))
        return intersection / union

    def get_prediction(self, sess, x_data, from_paths=False,
                       check_compatibility=False,
                       compatibility_multiplier=32,
                       tgt_size=None, method=None,
                       keep_prob=1.0):
        """ Prediction of the neural network graph. """
        # Load images if needed
        if from_paths:
            x_data = utils.load_images(
                x_data, tgt_size=tgt_size, method=method,
                check_compatibility=check_compatibility,
                compatibility_multiplier=compatibility_multiplier)

        # Do it one by one if different sizes
        if from_paths and method is None:
            pred = []
            for x in x_data:
                pred.append(
                    sess.run(tf.nn.softmax(self.logits_tf),
                             feed_dict={self.x_tf: np.expand_dims(x, axis=0),
                                        self.keep_prob_tf: keep_prob}))
                if pred[-1].shape[-1] == 2:
                    pred[-1] = pred[-1][:, :, :, 1]
        # Do it for all the batch
        else:
            pred = sess.run(tf.nn.softmax(self.logits_tf),
                            feed_dict={self.x_tf: x_data,
                                       self.keep_prob_tf: keep_prob})
            if pred.shape[-1] == 2:
                pred = pred[:, :, :, 1]

        return pred

    def load_pretrained_weights(self, sess):
        tf.train.Saver(self.weights).restore(sess, self.tf_checkpoint_path)

    def train_graph(self, sess,
                    x_train, y_train,
                    x_valid, y_valid,
                    w_train=None, w_valid=None,
                    method='resize',
                    n_epoch=1, train_on_augmented_data=False,
                    train_profille='all', lr=None):
        """ Train the graph of the corresponding neural network. """
        # Decide what optmizer to use
        assert train_profille in ['all', 'top']
        if train_profille == 'all':
            train_step = self.train_step_all_tf
        elif train_profille == 'top':
            train_step = self.train_step_top_tf

        # if lr, overwrite self.learn_rate and reset self.learn_rate_pos
        if lr is not None:
            self.learn_rate_0 = lr
            self.learn_rate_pos = 0
            self.epoch = 0
        # Set training and validation sets.
        self.x_train = x_train
        self.y_train = y_train

        self.x_valid = x_valid
        self.y_valid = y_valid

        self.w_train = w_train
        self.w_valid = w_valid

        # Parameters.
        self.perm_array = np.arange(len(self.x_train))
        self.train_on_augmented_data = train_on_augmented_data
        mb_per_epoch = self.x_train.shape[0] / self.mb_size
        tgt_size = self.input_shape[:2]

        # Start timer.
        start = datetime.datetime.now()
        print('Training the Neural Network')
        print('\tnn_name = {}, n_epoch = {}, mb_size = {}, learnrate = {:.7f}'.format(
            self.nn_name, n_epoch, self.mb_size, self.learn_rate))
        print('\tinput_shape = {}, output_shape = {}'.format(
            self.input_shape, self.output_shape))
        print('\tlearn_rate = {:.10f}, learn_rate_0 = {:.10f}, learn_rate_alpha = {}'.format(
            self.learn_rate, self.learn_rate_0, self.learn_rate_alpha))
        print('\tlearn_rate_step = {}, learn_rate_pos = {}, dropout_proba = {}'.format(
            self.learn_rate_step, self.learn_rate_pos, self.dropout_proba))
        print('\tx_train = {}, x_valid = {}'.format(
            x_train.shape, x_valid.shape))
        print('\ty_train = {}, y_valid = {}'.format(
            y_train.shape, y_valid.shape))
        print('Training started: {}'.format(datetime.datetime.now().strftime(
            '%d-%m-%Y %H:%M:%S')))

        # Looping over mini batches.
        for i in range(int(n_epoch * mb_per_epoch) + 1):

            # Adapt the learning rate.
            if not self.learn_rate_pos == int(self.epoch // self.learn_rate_step):
                self.learn_rate_pos += 1
                self.learn_rate = self.get_learn_rate()
                print('Update learning rate to {:.10f}. Running time: {}'.format(
                    self.learn_rate, datetime.datetime.now() - start))

            # Train the graph.
            x_batch, y_batch, w_batch = self.next_mini_batch(
                method, tgt_size)  # next mini batch
            sess.run([train_step, self.extra_update_ops_tf],
                     feed_dict={self.x_tf: x_batch,
                                self.y_tf: y_batch,
                                # self.w_tf: w_batch,
                                self.keep_prob_tf: self.keep_prob,
                                self.learn_rate_tf: self.learn_rate,
                                tf.keras.backend.learning_phase(): 1,  # Required if using tf.keras
                                self.training_tf: True})

            # Store losses and scores.
            if i % int(self.log_step * mb_per_epoch) == 0:

                self.n_log_step += 1  # Current number of log steps.

                trn_dct = dict(x=self.x_train, y=self.y_train, w=self.w_train)
                val_dct = dict(x=self.x_valid, y=self.y_valid, w=self.w_valid)
                for dct in [trn_dct, val_dct]:
                    # Random ids for eval (same size as val)
                    ids = np.arange(len(dct['x']))
                    np.random.shuffle(ids)
                    ids = ids[:len(val_dct['x'])]  # len(x_batch)
                    x = utils.load_images(dct['x'][ids],
                                          method='resize', tgt_size=tgt_size)
                    y = utils.load_masks(dct['y'][ids],
                                         method='resize', tgt_size=tgt_size)
                    # w = dct['w'][ids]

                    dct['feed_dict'] = {self.x_tf: x,
                                        self.y_tf: y,
                                        # self.w_tf: w,
                                        self.keep_prob_tf: 1.0}

                    # Evaluate current loss and score
                    dct['loss'] = sess.run(self.loss_tf,
                                           feed_dict=dct['feed_dict'])
                    pred = self.get_prediction(sess, x)
                    dct['score'] = np.mean(self.get_score(pred, y))
                    dct['iou'] = np.mean(self.get_IoU(pred, y))

                print(('{:.2f} epoch: train/valid loss = {:.4f}/{:.4f} ' +
                       'train/valid score = {:.4f}/{:.4f} ' +
                       'train/valid IoU = {:.4f}/{:.4f}').format(
                    self.epoch, trn_dct['loss'], val_dct['loss'],
                      trn_dct['score'], val_dct['score'],
                      trn_dct['iou'], val_dct['iou']))

                # Store losses and scores.
                self.params['train_loss'].extend([trn_dct['loss']])
                self.params['valid_loss'].extend([val_dct['loss']])
                self.params['train_score'].extend([trn_dct['score']])
                self.params['valid_score'].extend([val_dct['score']])
                self.params['train_iou'].extend([trn_dct['iou']])
                self.params['valid_iou'].extend([val_dct['iou']])

                # Save summaries for TensorBoard.
                if self.use_tb_summary:
                    train_summary = sess.run(
                        self.merged, feed_dict=trn_dct['feed_dict'])
                    valid_summary = sess.run(
                        self.merged, feed_dict=val_dct['feed_dict'])
                    self.train_writer.add_summary(
                        train_summary, self.n_log_step)
                    self.valid_writer.add_summary(
                        valid_summary, self.n_log_step)

        # Store parameters.
        self.params['learn_rate'] = self.learn_rate
        self.params['learn_rate_step'] = self.learn_rate_step
        self.params['learn_rate_pos'] = self.learn_rate_pos
        self.params['learn_rate_alpha'] = self.learn_rate_alpha
        self.params['learn_rate_0'] = self.learn_rate_0
        self.params['keep_prob'] = self.keep_prob
        self.params['epoch'] = self.epoch
        self.params['n_log_step'] = self.n_log_step
        self.params['log_step'] = self.log_step
        self.params['input_shape'] = self.input_shape
        self.params['output_shape'] = self.output_shape
        self.params['mb_size'] = self.mb_size
        self.params['dropout_proba'] = self.dropout_proba

        print('Training ended. Running time: {}'.format(
            datetime.datetime.now() - start))
