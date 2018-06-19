# Import necessary modules and set global constants and variables.
import os                          # For filepath, directory handling
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.model_selection     # For using KFold
import datetime                    # To measure running time
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling
import matplotlib.pyplot as plt    # Python 2D plotting library
import matplotlib.cm as cm         # Color map
import utils
import score
from NeuralNetworks import U_Net


# ##################### Global constants #################
IMG_WIDTH = 256        # Default image width
IMG_HEIGHT = 256       # Default image height
IMG_CHANNELS = 3       # Default number of channels
NET_TYPE = 'Xception_InceptionSE'  # Network to use
nn_name = 'unet_xception_256crops_wdice+bce'
PRETRAIN_WEIGHTS = False
USE_WEIGHTS = False    # For weighted bce
DATA_METHOD = 'crop'   # Either crop or resize

# %%####################### DIRS #########################
TRAIN_DIR = os.path.join(os.getcwd(), 'stage1_train')
TEST_DIR = os.path.join(os.getcwd(), 'stage1_final_test')
#TRAIN_DIR = os.path.join(os.getcwd(), 'External datasets/external_data/train')
#TEST_DIR = os.path.join(os.getcwd(), 'External datasets/external_data/test')
IMG_TYPE = '.png'         # Image type
DIR_DICT = dict(logs='logs', saves='saves')
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks
SEED = 123                # Random seed for splitting train/validation sets

# #####################Global variables####################
min_object_size = 20       # Minimal nucleous size in pixels
x_train = []
y_train = []
x_test = []
y_test_pred_proba = {}
y_test_pred = {}

# %%##################### HYPERPARAMETERS ##################
LEARN_RATE_0 = 0.01
LEARN_RATE_ALPHA = 0.25
LEARN_RATE_STEP = 3
N_EPOCH = 100
MB_SIZE = 10
USE_BN = False
USE_DROP = False
KEEP_PROB = 0.8
ACTIVATION = 'selu'
PADDING = 'SYMMETRIC'
LOSS = 'wdice+ce+entropy_penalty'

# %%############ LOADING DATASETS #############################
# Display working/train/test directories.
print('CW_DIR = {}'.format(os.getcwd()))
print('TRAIN_DIR = {}'.format(TRAIN_DIR))
print('TEST_DIR = {}'.format(TEST_DIR))


# Basic properties of images/masks.
train_df = utils.read_train_data_properties(TRAIN_DIR,
                                            IMG_DIR_NAME,
                                            MASK_DIR_NAME)
test_df = utils.read_test_data_properties(TEST_DIR, IMG_DIR_NAME)
print('train_df:')
print(train_df.describe())
print('')
print('test_df:')
print(test_df.describe())

# Counting unique image shapes.
df = pd.DataFrame([[x]
                   for x in zip(train_df['img_height'],
                                train_df['img_width'])])
df[0].value_counts()


# Get rerefence for image and mask paths
x_train, y_train = train_df['image_path'].values, train_df['mask_dir'].values


# %%#####################################################################
# ############################# TRAINING ################################
# #######################################################################
# Implement cross validations
cv_num = 10
kfold = sklearn.model_selection.KFold(
    cv_num, shuffle=True, random_state=SEED)

for i, (train_index, valid_index) in enumerate(kfold.split(x_train)):
    # Start timer
    start = datetime.datetime.now()

    # Split into train and validation
    x_trn = x_train[train_index]
    y_trn = y_train[train_index]

    x_vld = x_train[valid_index]
    y_vld = y_train[valid_index]

    # Choose a certain fold.
    if i == 0:
        # Create and start training of a new model.
        if not PRETRAIN_WEIGHTS:
            u_net = U_Net(nn_name=nn_name, mb_size=MB_SIZE, log_step=1.0,
                          input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                          output_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
                          learn_rate_0=LEARN_RATE_0,
                          learn_rate_alpha=LEARN_RATE_ALPHA,
                          learn_rate_step=LEARN_RATE_STEP,
                          activation_fun=ACTIVATION,
                          padding=PADDING,
                          loss=LOSS,
                          use_bn=USE_BN, use_drop=USE_DROP,
                          use_weights=USE_WEIGHTS,
                          net_type=NET_TYPE,
                          dir_dict=DIR_DICT,
                          )
            u_net.build_graph()  # Build graph.

            # Start tensorflow session.
            with tf.Session(graph=u_net.graph) as sess:
                u_net.attach_saver()  # Attach saver tensor.
                u_net.attach_summary(sess)  # Attach summaries.
                # Variable initialization.
                sess.run(tf.global_variables_initializer())
                u_net.load_pretrained_weights(sess)

#                # Training on original data.
                u_net.train_graph(sess,
                                  x_train=x_trn, y_train=y_trn,
                                  x_valid=x_vld, y_valid=y_vld,
                                  n_epoch=1.,
                                  train_profille='top',
                                  method=DATA_METHOD,
                                  )
               # Training on augmented data.
                u_net.train_graph(sess,
                                  x_train=x_trn, y_train=y_trn,
                                  x_valid=x_vld, y_valid=y_vld,
                                  n_epoch=20,
                                  train_on_augmented_data=False,
                                  train_profille='top',
                                  method=DATA_METHOD,
                                  )
                #                u_net.learn_rate_alpha = 0.15
                u_net.train_graph(sess,
                                  x_train=x_trn, y_train=y_trn,
                                  x_valid=x_vld, y_valid=y_vld,
                                  n_epoch=N_EPOCH,
<<<<<<< HEAD
                                  train_on_augmented_data=True,
                                      #                                   lr = 0.0001,
                                  train_profille='all',
                                  method=DATA_METHOD,
                                  )
=======
                                  train_on_augmented_data=False,
                                  #                                   lr = 0.0001,
                                  train_profille='all')
>>>>>>> master

                # Save parameters, tensors, summaries.
                u_net.save_model(sess)

        # Continue training of a pretrained model.
        else:
            u_net = U_Net()
            sess = u_net.load_session_from_file(nn_name,
                                                update_cost=True,
                                                renew_LR=False)
            u_net.attach_saver()
            u_net.attach_summary(sess)

            # Training on augmented data.
            u_net.train_graph(sess,
                              x_train=x_trn, y_train=y_trn,
                              x_valid=x_vld, y_valid=y_vld,
                              n_epoch=N_EPOCH,
                              train_on_augmented_data=True)
            u_net.save_model(sess)  # Save parameters, tensors, summaries.

print('Total running time: ', datetime.datetime.now() - start)

# %%Show intermediate losses and scores during the training session.
u_net = U_Net(dir_dict=DIR_DICT)
sess = u_net.load_session_from_file(nn_name)
sess.close()
train_loss = u_net.params['train_loss']
valid_loss = u_net.params['valid_loss']
train_score = u_net.params['train_score']
valid_score = u_net.params['valid_score']

print(
    'final train/valid loss = {:.4f}/{:.4f}'.format(
        train_loss[-1], valid_loss[-1]))
print(
    'final train/valid score = {:.4f}/{:.4f}'.format(
        train_score[-1], valid_score[-1]))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, len(train_loss)), train_loss, '-b', label='Training')
plt.plot(np.arange(0, len(valid_loss)),
         valid_loss, '-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax=0.5, ymin=0.0)
plt.ylabel('loss')
plt.xlabel('steps')

plt.subplot(1, 2, 2)
plt.plot(np.arange(0, len(train_score)),
         train_score, '-b', label='Training')
plt.plot(np.arange(0, len(valid_score)),
         valid_score, '-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax=1.0, ymin=0.0)
plt.ylabel('score')
plt.xlabel('steps')

# %% ####### SAME K-fold in case training is not run #########
cv_num = 10
kfold = sklearn.model_selection.KFold(
    cv_num, shuffle=True, random_state=SEED)

for i, (train_index, valid_index) in enumerate(kfold.split(x_train)):
    # Start timer
    start = datetime.datetime.now()

    # Split into train and validation
    x_trn = x_train[train_index]
    y_trn = y_train[train_index]
    w_trn = y_weights[train_index]

    x_vld = x_train[valid_index]
    y_vld = y_train[valid_index]
    w_vld = y_weights[valid_index]

# %%Summary of scores for training and validations sets. Note that the score is
# better than the true score, since overlapping/touching nuclei can not be
# separately identified in this version.
u_net = U_Net(dir_dict=DIR_DICT)
sess = u_net.load_session_from_file(nn_name)

# Overall score on validation set.
y_valid_pred_proba = u_net.get_prediction(sess, x_vld)
for i in range(len(y_valid_pred_proba)):
    y_valid_pred_proba[i] = y_valid_pred_proba[i] / y_valid_pred_proba[i].max()
y_valid_pred = utils.trsf_proba_to_binary(
    y_valid_pred_proba, threshold=0.5)
valid_score = u_net.get_score(y_valid_pred_proba, y_vld)
tmp = np.concatenate([np.arange(len(valid_index)).reshape(-1, 1),
                      valid_index.reshape(-1, 1),
                      valid_score.reshape(-1, 1)], axis=1)
valid_score_df = pd.DataFrame(tmp, columns=(
    ['index', 'valid_index', 'valid_score']))
print('\n', valid_score_df.describe())
print('\n', valid_score_df.sort_values(
    by='valid_score', ascending=True).head())

# Plot the worst 4 predictions.
fig, axs = plt.subplots(4, 4, figsize=(20, 20))
list_ = valid_score_df.sort_values(by='valid_score', ascending=True)[
    :4]['index'].values.astype(np.int)
# list_ = [valid_score_df['valid_score'].idxmin(),valid_score_df['valid_score'].idxmax()]
for i, n in enumerate(list_):
    img = utils.imshow_args(x_vld[n])
    axs[i, 0].imshow(img)
    axs[i, 0].set_title('{}.) input image'.format(n))
    axs[i, 1].imshow(np.squeeze(y_vld[n]), cmap='jet')
    axs[i, 1].set_title('{}.) true mask'.format(n))
    axs[i, 2].imshow(y_valid_pred_proba[n], cmap='jet')
    axs[i, 2].set_title('{}.) predicted mask probabilities'.format(n))
    axs[i, 3].imshow(y_valid_pred[n], cmap='jet')
    axs[i, 3].set_title('{}.) predicted mask'.format(n))

sess.close()
#%%
if False:

    # In[ ]:

    # Tune minimal object size for prediction
    if True:
        # mn = 'nn0_512_512_3'
        mn = nn_name[0]
        u_net = NeuralNetwork()
        sess = u_net.load_session_from_file(mn)
        y_valid_pred_proba = u_net.get_prediction(sess, x_vld)
        y_valid_pred = trsf_proba_to_binary(y_valid_pred_proba, threshold=0.5)
        sess.close()

        tmp = min_object_size
        min_object_sizes = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70,
                            80, 90, 100, 110, 120, 130, 140, 150, 200, 300, 400, 500]
        for mos in min_object_sizes:
            min_object_size = mos
            valid_score = get_score(y_vld, y_valid_pred)
            print('min_object_size = {}: valid_score min/mean/std/max = {:.3f}/{:.3f}/{:.3f}/{:.3f}'.format(mos,
                                                                                                            np.min(valid_score), np.mean(valid_score), np.std(valid_score), np.max(valid_score)))
        min_object_size = tmp

    # In[ ]:

    # Check one sample prediction in more detail.
    # mn = 'nn0_512_512_3'
    mn = nn_name[0]
    u_net = NeuralNetwork()
    sess = u_net.load_session_from_file(mn)
    n = np.random.randint(len(x_vld))
    x_true = x_vld[n]
    y_true = y_vld[n, :, :, 0]
    y_pred_proba = u_net.get_prediction(
        sess, np.expand_dims(x_true, axis=0))[0, :, :, 0]
    y_pred = trsf_proba_to_binary(y_pred_proba, threshold=0.5)
    sess.close()

    fig, axs = plt.subplots(1, 3, figsize=(20, 13))
    img, img_type = imshow_args(x_true)
    axs[0].imshow(img, img_type)
    axs[0].set_title('{}.) input image'.format(n))
    axs[1].imshow(y_pred_proba, cmap='gray')
    axs[1].set_title('{}.) predicted mask probabilities'.format(n))
    axs[2].imshow(y_pred, cmap='gray')
    axs[2].set_title('{}.) predicted mask'.format(n))
    plot_score_summary(y_true, y_pred)

    # # 8. Make Test Prediction <a class="anchor" id="8-bullet"></a>

    # In[10]:

    # Collection of methods for run length encoding.
    # For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included
    # in the mask. The pixels are one-indexed and numbered from top to bottom,
    # then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

    def rle_of_binary(x):
        """ Run length encoding of a binary 2D array. """
        dots = np.where(x.T.flatten() == 1)[0]  # indices from top to down
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def mask_to_rle(mask, cutoff=.5, min_object_size=20):
        """ Return run length encoding of mask. """
        # segment image and label different objects
        lab_mask = skimage.morphology.label(mask > cutoff)

        # Keep only objects that are large enough.
        (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
        if (mask_sizes < min_object_size).any():
            mask_labels = mask_labels[mask_sizes < min_object_size]
            for n in mask_labels:
                lab_mask[lab_mask == n] = 0
            lab_mask = skimage.morphology.label(lab_mask > cutoff)

        # Loop over each object excluding the background labeled by 0.
        for i in range(1, lab_mask.max() + 1):
            yield rle_of_binary(lab_mask == i)

    def rle_to_mask(rle, img_shape):
        ''' Return mask from run length encoding.'''
        mask_rec = np.zeros(img_shape).flatten()
        for n in range(len(rle)):
            for i in range(0, len(rle[n]), 2):
                for j in range(rle[n][i + 1]):
                    mask_rec[rle[n][i] - 1 + j] = 1
        return mask_rec.reshape(img_shape[1], img_shape[0]).T

    # In[11]:

    # Load neural network, make prediction for test masks, resize predicted
    # masks to original image size and apply run length encoding for the
    # submission file.

    # Load neural network and make prediction for masks.
    # nn_name = ['nn0_512_512_3']
    # nn_name = ['nn0_384_384_3_res']

    # Soft voting majority.
    inference_batch = 200
    count, rest = divmod(len(x_test), inference_batch)
    for i, mn in enumerate(nn_name):
        u_net = NeuralNetwork()
        sess = u_net.load_session_from_file(mn)
        y_test_pred_proba = []
        j = 0
        for k in range(count):
            if i == 0:
                pred = u_net.get_prediction(
                    sess, x_test[j:j + inference_batch]) / len(nn_name)
            else:
                pred += u_net.get_prediction(sess,
                                             x_test[j:j + inference_batch]) / len(nn_name)
            j += inference_batch
            y_test_pred_proba.extend(pred)

        if rest:
            if i == 0:
                pred = u_net.get_prediction(sess, x_test[j:]) / len(nn_name)
            else:
                pred += u_net.get_prediction(sess, x_test[j:]) / len(nn_name)
        y_test_pred_proba.extend(pred)
        print(len(y_test_pred_proba))
        sess.close()

    y_test_pred = trsf_proba_to_binary(y_test_pred_proba)[:, :, :, 0]

    print('y_test_pred.shape = {}'.format(y_test_pred.shape))

    # Resize predicted masks to original image size.
    y_test_pred_original_size = []
    for i in range(len(y_test_pred)):
        res_mask = trsf_proba_to_binary(skimage.transform.resize(np.squeeze(y_test_pred[i]),
                                                                 (test_df.loc[i, 'img_height'], test_df.loc[
                                                                  i, 'img_width']),
                                                                 mode='constant', preserve_range=True))
        y_test_pred_original_size.append(res_mask)
    y_test_pred_original_size = np.array(y_test_pred_original_size)

    print('y_test_pred_original_size.shape = {}'.format(
        y_test_pred_original_size.shape))

    # Run length encoding of predicted test masks.
    test_pred_rle = []
    test_pred_ids = []
    for n, id_ in enumerate(test_df['img_id']):
        min_object_size = 20 * test_df.loc[n, 'img_height'] * test_df.loc[n, 'img_width'] / (IMG_WIDTH *
                                                                                             IMG_HEIGHT)
        rle = list(mask_to_rle(
            y_test_pred_original_size[n], min_object_size=50))
        test_pred_rle.extend(rle)
        test_pred_ids.extend([id_] * len(rle))

    print('test_pred_ids.shape = {}'.format(np.array(test_pred_ids).shape))
    print('test_pred_rle.shape = {}'.format(np.array(test_pred_rle).shape))

    # In[32]:

    # Inspect a test prediction and check run length encoding.
    # n = np.random.randint(len(x_test))
    n = 921
    mask = y_test_pred_original_size[n]
    rle = list(mask_to_rle(mask))
    mask_rec = rle_to_mask(rle, mask.shape)
    print('Run length encoding: {} matches, {} misses'.format(
        (mask_rec == mask).sum(), (mask_rec != mask).sum()))

    fig, axs = plt.subplots(2, 3, figsize=(20, 13))
    axs[0, 0].imshow(read_image(test_df['image_path'].loc[n]))
    axs[0, 0].set_title('{}.) original test image'.format(n))
    img, img_type = imshow_args(x_test[n])
    axs[0, 1].imshow(img, img_type)
    axs[0, 1].set_title('{}.) transformed test image'.format(n))
    axs[0, 2].imshow(y_test_pred_proba[n][:, :, 0], cm.gray)
    axs[0, 2].set_title('{}.) predicted test mask probabilities'.format(n))
    axs[1, 0].imshow(y_test_pred_proba[n][:, :, 0], cm.gray)
    axs[1, 0].set_title('{}.) predicted test mask'.format(n))
    axs[1, 1].imshow(y_test_pred_original_size[n], cm.gray)
    axs[1, 1].set_title(
        '{}.) predicted final test mask in original size'.format(n))
    axs[1, 2].imshow(mask_rec[:, :], cm.gray)
    axs[1, 2].set_title(
        '{}.) final mask recovered from run length encoding'.format(n))

    # # 9. Submit <a class="anchor" id="9-bullet"></a>

    # In[33]:

    # Create submission file
    sub = pd.DataFrame()
    sub['ImageId'] = test_pred_ids
    sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(
        lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018-1.csv', index=False)
    sub.head()
