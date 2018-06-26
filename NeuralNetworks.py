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
        self.params['train_mask_iou'] = []
        self.params['valid_mask_iou'] = []
        self.params['train_mask2_iou'] = []
        self.params['valid_mask2_iou'] = []
        self.params['train_border_iou'] = []
        self.params['valid_border_iou'] = []

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

    def __init__(self, *args, loss='ce', net_type='vanilla', multi_head=False, ** kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.net_type = net_type
        self.multi_head = multi_head
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
            logits = self.create_graph()
            self.full_mask_logits_tf = tf.identity(
                logits[0], name='full_mask_logits_tf')
            if self.multi_head:
                self.borders_logits_tf = tf.identity(
                    logits[1], name='borders_logits_tf')

            # Target tensor.
            shape = [None, None, None, None]
            # shape.extend(self.output_shape)
            self.y_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='y_tf')  # (.,128,128,1)
            self.full_mask_y_tf = tf.identity(
                self.y_tf[:, :, :, :1], name='full_mask_y_tf')
            if self.multi_head:
                self.borders_y_tf = tf.identity(
                    self.y_tf[:, :, :, 1:], name='border_y_tf')

            # Weights tensor.
            shape = [None, self.input_shape[0], self.input_shape[1], 1]
            self.w_tf = tf.placeholder(dtype=tf.float32, shape=shape,
                                       name='w_tf')  # (.,128,128,1)

            # Loss tensor
            if self.loss[0] is not None:
                loss = tf.reduce_sum([l(self.full_mask_logits_tf, self.full_mask_y_tf)
                                      for l in self.loss[0]])
            else:
                loss = 0
            if self.multi_head:
                loss = loss + tf.reduce_sum(
                    [l(self.borders_logits_tf, self.borders_y_tf)
                     for l in self.loss[1]])

            self.loss_tf = tf.identity(loss, name='loss_tf')

            # Optimization tensor.
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
                              out_shape=[2, 3],
                              multi_head=self.multi_head,
                              padding=self.padding)
        else:
            if self.net_type == 'Xception_vanilla':
                logits, self.encoder, self.end_points = \
                    unets.Xception_vanilla(self.x_tf,
                                           activation_fn=self.activation_fun,
                                           out_shape=[2, 3],
                                           multi_head=self.multi_head,
                                           padding=self.padding)

            elif self.net_type == 'SE_Xception_vanillaSE':
                logits, self.encoder, self.end_points = \
                    unets.SE_Xception_vanillaSE(self.x_tf,
                                                activation_fn=self.activation_fun,
                                                out_shape=[2, 3],
                                                multi_head=self.multi_head,
                                                padding=self.padding)

            elif self.net_type == 'Xception_InceptionSE':
                logits, self.encoder, self.end_points = \
                    unets.Xception_InceptionSE(self.x_tf,
                                               activation_fn=self.activation_fun,
                                               out_shape=[2, 3],
                                               multi_head=self.multi_head,
                                               padding=self.padding)

            elif self.net_type == 'InceptionResNetV2_vanilla':
                logits, self.encoder, self.end_points = \
                    unets.InceptionResNetV2(self.x_tf,
                                            activation_fn=self.activation_fun,
                                            out_shape=[2, 3],
                                            multi_head=self.multi_head,
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
        self.full_mask_logits_tf = graph.get_tensor_by_name(
            "full_mask_logits_tf:0")
        if self.multi_head:
            self.borders_y_tf = graph.get_tensor_by_name("border_y_tf:0")
            self.borders_logits_tf = graph.get_tensor_by_name(
                "borders_logits_tf:0")
        if not update_cost:
            self.loss_tf = graph.get_tensor_by_name('loss_tf:0')
        else:
            self.loss_tf = self.loss_tensor()
        self.training_tf = graph.get_tensor_by_name("training_tf:0")
        self.extra_update_ops_tf = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

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
        self.multi_head = self.params['multi_head']

        print('Parameters of the loaded neural network')
        print('\tnn_name = {}, epoch = {:.2f}, mb_size = {}'.format(
            self.nn_name, self.epoch, self.mb_size))
        print('\tinput_shape = {}, output_shape = {}'.format(
            self.input_shape, self.output_shape))
        print('\tlearn_rate = {:.10f}, learn_rate_0 = {:.10f}, dropout_proba = {}'.format(
            self.learn_rate, self.learn_rate_0, self.dropout_proba))
        print('\tlearn_rate_step = {}, learn_rate_pos = {}, learn_rate_alpha = {}'.format(
            self.learn_rate_step, self.learn_rate_pos, self.learn_rate_alpha))

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
            self.y_train[self.perm_array[start:end]])

        # Use augmented data.
        if self.train_on_augmented_data:
            x_tr, y_tr = utils.augment_images_masks(
                x_tr, y_tr)
            y_tr = utils.trsf_proba_to_binary(y_tr)

        return x_tr, y_tr, None

    def get_score(self, pred, y, from_paths=False, tgt_size=None, post_processing=False, method=None):
        '''Reimplement this function and return the score'''
        # for i in range(len(pred)):
        #     pred[i] = pred[i] / pred[i].max()
        if from_paths:
            y = utils.load_masks(y, tgt_size=tgt_size, method=method)
        if not post_processing:
            pred = utils.trsf_proba_to_binary(pred)
        s = score.get_score(y, pred, label_pred=(
            self.multi_head and post_processing))
        return s

    def get_IoU(self, pred, y):
        pred = utils.trsf_proba_to_binary(pred)
        if len(pred.shape) > 3:
            pred = np.squeeze(pred)

        intersection = np.sum(pred * y)
        union = np.sum(pred + y)
        return 2 * intersection / union

    def get_prediction(self, sess, x_data,
                       tgt_size=None, method=None,
                       full_prediction=False,
                       keep_prob=1.0):

        pred = sess.run(tf.nn.softmax(self.full_mask_logits_tf),
                        feed_dict={self.x_tf: x_data,
                                   self.keep_prob_tf: keep_prob})

        if self.multi_head and full_prediction:
            borders = sess.run(tf.nn.softmax(self.borders_logits_tf),
                               feed_dict={self.x_tf: x_data,
                                          self.keep_prob_tf: keep_prob})
            pred = np.concatenate(
                [pred[:, :, :, 1:], borders[:, :, :, 1:]], -1)

        else:
            pred = pred[:, :, :, 1:]

        return pred

    def get_prediction_from_path(self, sess, x_data,
                                 compatibility_multiplier=32,
                                 tgt_size=None, method=None,
                                 full_prediction=False,
                                 keep_prob=1.0):
        """ Prediction of the neural network graph. """
        # Load images
        x_data = utils.load_images(
            x_data, tgt_size=tgt_size, method=method)

        # Do it one by one if different sizes
        if method is None:
            pred = []
            for x in x_data:
                # Pad if required
                x, pads = utils.match_size_with_pad(
                    x, compatibility_multiplier)
                x = np.expand_dims(x, axis=0)
                # Run for full mask
                p = sess.run(tf.nn.softmax(self.full_mask_logits_tf),
                             feed_dict={self.x_tf: x,
                                        self.keep_prob_tf: keep_prob})
                if self.multi_head and full_prediction:
                    # Run for borders and untouching mask
                    b = sess.run(tf.nn.softmax(self.borders_logits_tf),
                                 feed_dict={self.x_tf: x,
                                            self.keep_prob_tf: keep_prob})

                    p = np.concatenate([p[0, :, :, 1:], b[0, :, :, 1:]], -1)
                else:
                    p = p[0, :, :, 1]
                # Unpad
                pred.append(utils.unpad_image_to_original_size(
                    p, pads))

        # Do it for all the batch
        else:
            pred = sess.run(tf.nn.softmax(self.full_mask_logits_tf),
                            feed_dict={self.x_tf: x_data,
                                       self.keep_prob_tf: keep_prob})

            if self.multi_head and full_prediction:
                borders = sess.run(tf.nn.softmax(self.borders_logits_tf),
                                   feed_dict={self.x_tf: x_data,
                                              self.keep_prob_tf: keep_prob})
                pred = np.concatenate(
                    [pred[:, :, :, 1:], borders[:, :, :, 1:]], -1)

            else:
                pred = pred[:, :, :, 1:]

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

                trn_dct = dict(x=self.x_train,
                               y=self.y_train, w=self.w_train)
                val_dct = dict(x=self.x_valid,
                               y=self.y_valid, w=self.w_valid)
                for dct in [trn_dct, val_dct]:
                    # Random ids for eval (same size as val)
                    ids = np.arange(len(dct['x']))
                    np.random.shuffle(ids)
                    ids = ids[:len(val_dct['x'])]  # len(x_batch)
                    x, y = utils.load_images_masks(
                        dct['x'][ids], dct['y'][ids])
                    # w = dct['w'][ids]

                    dct['feed_dict'] = {self.x_tf: x,
                                        self.y_tf: y,
                                        # self.w_tf: w,
                                        self.keep_prob_tf: 1.0}

                    # Evaluate current loss and score
                    dct['loss'] = sess.run(self.loss_tf,
                                           feed_dict=dct['feed_dict'])
                    pred = self.get_prediction(sess, x, full_prediction=True)
                    dct['mask_iou'] = np.mean(
                        self.get_IoU(pred[:, :, :, 0], y[:, :, :, 0]))
                    if self.multi_head:
                        dct['border_iou'] = np.mean(
                            self.get_IoU(pred[:, :, :, 2], y[:, :, :, 3]))
                        dct['mask2_iou'] = np.mean(
                            self.get_IoU(pred[:, :, :, 1], y[:, :, :, 2]))
                    else:
                        dct['mask2_iou'] = 0
                        dct['border_iou'] = 0

                print(('{:.2f} epoch: train/valid loss = {:.4f}/{:.4f} ' +
                       'train/valid mask IoU = {:.4f}/{:.4f} ' +
                       'train/valid borderless mask IoU = {:.4f}/{:.4f} ' +
                       'train/valid border IoU = {:.4f}/{:.4f}').format(
                    self.epoch, trn_dct['loss'], val_dct['loss'],
                      trn_dct['mask_iou'], val_dct['mask_iou'],
                      trn_dct['mask2_iou'], val_dct['mask2_iou'],
                      trn_dct['border_iou'], val_dct['border_iou']))

                # Store losses and scores.
                self.params['train_loss'].extend([trn_dct['loss']])
                self.params['valid_loss'].extend([val_dct['loss']])
                self.params['train_mask_iou'].extend([trn_dct['mask_iou']])
                self.params['valid_mask_iou'].extend([val_dct['mask_iou']])
                self.params['train_mask2_iou'].extend([trn_dct['mask2_iou']])
                self.params['valid_mask2_iou'].extend([val_dct['mask2_iou']])
                self.params['train_border_iou'].extend([trn_dct['border_iou']])
                self.params['valid_border_iou'].extend([val_dct['border_iou']])

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
        self.params['multi_head'] = self.multi_head

        print('Training ended. Running time: {}'.format(
            datetime.datetime.now() - start))
