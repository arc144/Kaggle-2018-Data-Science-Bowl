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
from loss import categorical_cross_entropy, soft_dice
from tqdm import tqdm
from NeuralNetworks import U_Net
from scipy.ndimage import label

# ##################### Global constants #################
IMG_WIDTH = 256        # Default image width
IMG_HEIGHT = 256       # Default image height
IMG_CHANNELS = 3       # Default number of channels
NET_TYPE = 'Xception_InceptionSE'  # Network to use
nn_name = 'unet_Xception_InceptionSE_V9_Aggaug_dice+bce_multihead'
USE_WEIGHTS = False    # For weighted bce
METHOD = 'resize'   # Either crop or resize
MULTI_HEAD = True
POST_PROCESSING = True
DATASET = 'V9'

# %%####################### DIRS #########################
if DATASET == 'V1':
    TRAIN_DIR = os.path.join(os.getcwd(), 'stage1_train')
elif DATASET == 'V9':
    TRAIN_DIR = os.path.join(
        os.getcwd(), 'External datasets/external_data/train')

TEST_DIR = os.path.join(os.getcwd(), 'stage2_final_test')
VAL_DIR = os.path.join(os.getcwd(), 'stage1_test')

# TEST_DIR = os.path.join(os.getcwd(), 'External datasets/external_data/test')
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

# %%###########################################################################
########################### HYPERPARAMETERS ###################################
###############################################################################
LEARN_RATE_0 = 1e-3
LEARN_RATE_FINETUNE = 1e-4
LEARN_RATE_ALPHA = 0.25
LEARN_RATE_STEP = 3
N_EPOCH = 30
MB_SIZE = 10
KEEP_PROB = 0.8
ACTIVATION = 'selu'
PADDING = 'SYMMETRIC'
AUGMENTATION = True

# LOSS = [[categorical_cross_entropy(), soft_dice(
#     logits_axis=1, label_axis=0, weight=2)]]

LOSS = [[categorical_cross_entropy(), soft_dice(logits_axis=1, label_axis=0)],
        [categorical_cross_entropy(onehot_convert=False),
         soft_dice(logits_axis=1, label_axis=1, weight=2),
         soft_dice(logits_axis=2, label_axis=2, weight=10)]]

# LOSS = [None,
#         [categorical_cross_entropy(onehot_convert=False),
#          soft_dice(logits_axis=1, label_axis=1),
#          soft_dice(logits_axis=2, label_axis=2, weight=2)]]

# %%###########################################################################
############################## LOADING DATASETS ###############################
###############################################################################
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
x_test, test_ids = test_df['image_path'].values, test_df['img_id'].values
test_sizes = [(h, w) for h, w in zip(
    test_df['img_height'].values, test_df['img_width'].values)]


# %%###########################################################################
# ################################ TRAINING ###################################
# #############################################################################
PRETRAIN_WEIGHTS = False
# Implement cross validations
if DATASET == 'V1':
    cv_num = 10
elif DATASET == 'V9':
    cv_num = 50

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
                          multi_head=MULTI_HEAD,
                          learn_rate_0=LEARN_RATE_0,
                          learn_rate_alpha=LEARN_RATE_ALPHA,
                          learn_rate_step=LEARN_RATE_STEP,
                          activation_fun=ACTIVATION,
                          padding=PADDING,
                          loss=LOSS,
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
                if NET_TYPE != 'vanilla':
                    u_net.load_pretrained_weights(sess)

    #                # Training on original data.
                    u_net.train_graph(sess,
                                      x_train=x_trn, y_train=y_trn,
                                      x_valid=x_vld, y_valid=y_vld,
                                      n_epoch=1.,
                                      train_profille='top',
                                      method='resize',
                                      )
                   # Training on augmented data.
                    u_net.train_graph(sess,
                                      x_train=x_trn, y_train=y_trn,
                                      x_valid=x_vld, y_valid=y_vld,
                                      n_epoch=5,
                                      train_on_augmented_data=AUGMENTATION,
                                      train_profille='top',
                                      method='resize',
                                      )
                #                u_net.learn_rate_alpha = 0.15
                u_net.train_graph(sess,
                                  x_train=x_trn, y_train=y_trn,
                                  x_valid=x_vld, y_valid=y_vld,
                                  n_epoch=N_EPOCH,
                                  train_on_augmented_data=AUGMENTATION,
                                  lr=LEARN_RATE_FINETUNE,
                                  train_profille='all',
                                  method='resize',
                                  )

                # Save parameters, tensors, summaries.
                u_net.save_model(sess)

        # Continue training of a pretrained model.
        else:
            u_net = U_Net(dir_dict=DIR_DICT)
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
                              train_on_augmented_data=True,
                              train_profille='all',
                              method='resize',
                              )
            u_net.save_model(sess)  # Save parameters, tensors, summaries.

print('Total running time: ', datetime.datetime.now() - start)

# %%###########################################################################
#### Show intermediate losses and scores during the training session.#######
############################################################################
u_net = U_Net(dir_dict=DIR_DICT)
sess = u_net.load_session_from_file(nn_name)
sess.close()
train_loss = u_net.params['train_loss']
valid_loss = u_net.params['valid_loss']
train_mask_iou = u_net.params['train_mask_iou']
valid_mask_iou = u_net.params['valid_mask_iou']
train_mask2_iou = u_net.params['train_mask2_iou']
valid_mask2_iou = u_net.params['valid_mask2_iou']
train_border_iou = u_net.params['train_border_iou']
valid_border_iou = u_net.params['valid_border_iou']

print(
    'final train/valid loss = {:.4f}/{:.4f}'.format(
        train_loss[-1], valid_loss[-1]))
print(
    'final train/valid mask iou = {:.4f}/{:.4f}'.format(
        train_mask_iou[-1], valid_mask_iou[-1]))
print(
    'final train/valid borderless mask iou = {:.4f}/{:.4f}'.format(
        train_mask2_iou[-1], valid_mask2_iou[-1]))
print(
    'final train/valid border iou = {:.4f}/{:.4f}'.format(
        train_border_iou[-1], valid_border_iou[-1]))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, len(train_loss)), train_loss, '-b', label='Training')
plt.plot(np.arange(0, len(valid_loss)),
         valid_loss, '-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
#plt.ylim(ymax=0.5, ymin=0.0)
plt.ylabel('loss')
plt.xlabel('steps')

plt.subplot(1, 2, 2)
plt.plot(np.arange(0, len(train_mask_iou)),
         train_mask_iou, '-b', label='Train. Mask IoU')
plt.plot(np.arange(0, len(valid_mask_iou)),
         valid_mask_iou, '-g', label='Val. Mask IoU')
plt.plot(np.arange(0, len(train_mask_iou)),
         train_mask2_iou, '-.b', label='Train. BorderlessMask IoU')
plt.plot(np.arange(0, len(valid_mask_iou)),
         valid_mask2_iou, '-.g', label='Val. BorderlessMask IoU')
plt.plot(np.arange(0, len(train_border_iou)),
         train_border_iou, ':b', label='Train. Border IoU')
plt.plot(np.arange(0, len(valid_border_iou)),
         valid_border_iou, ':g', label='Val. Border IoU')

plt.legend(loc='lower right', frameon=False)
#plt.ylim(ymax=1.0, ymin=0.0)
plt.ylabel('iou')
plt.xlabel('steps')


# %% ############## Check multi head precictions ##############################
u_net = U_Net(dir_dict=DIR_DICT)
sess = u_net.load_session_from_file(nn_name)
n = 47

pred = u_net.get_prediction_from_path(
    sess, [x_test[n]],
    method=METHOD,
    tgt_size=(IMG_HEIGHT, IMG_WIDTH),
    compatibility_multiplier=32,
    full_prediction=True)
pred = utils.trsf_proba_to_binary(np.array(pred))
markers = label(pred[0, :, :, 1] * (1 - pred[0, :, :, 2]))[0]
pred_water = utils.postprocessing(pred, method='watershed')
fig, axs = plt.subplots(1, 5, sharex=False)
axs[0].imshow(utils.read_image(x_test[n]))
axs[0].set_title('image')
axs[1].imshow(pred[0, :, :, 0], cmap='jet')
axs[1].set_title('pred full mask')
axs[2].imshow(pred[0, :, :, 2], cmap='jet')
axs[2].set_title('pred borders')
axs[3].imshow(markers, cmap='jet')
axs[3].set_title('pred markers')
axs[4].imshow(pred_water[0], cmap='jet')
axs[4].set_title('pred watershed')

sess.close()

# %%###########################################################################
# ########## Load neural network, make prediction for test and apply ##########
# ########## run length encoding for the submission file.            ##########
# #############################################################################
MIN_OBJECT_SIZE = 20
inference_batch = 50
count = len(x_test) // inference_batch
u_net = U_Net(dir_dict=DIR_DICT)
test_pred_rle = []
test_pred_ids = []

sess = u_net.load_session_from_file(nn_name)
i = 0
for j in tqdm(range(0, (count + 1) * inference_batch, inference_batch)):
    ids = test_ids[j:j + inference_batch]
    data = x_test[j:j + inference_batch]
    sizes = test_sizes[j:j + inference_batch]
    y_test_pred = u_net.get_prediction_from_path(
        sess, data,
        method=METHOD,
        tgt_size=(IMG_HEIGHT, IMG_WIDTH),
        compatibility_multiplier=32,
        full_prediction=True)

    y_test_pred = utils.trsf_proba_to_binary(y_test_pred)

    if METHOD == 'resize':
        y_test_pred = utils.resize_as_original(y_test_pred, sizes)

    if POST_PROCESSING:
        y_test_pred = utils.postprocessing(y_test_pred, method='watershed')

    rle, rle_id = utils.mask_to_rle_wrapper(
        y_test_pred, ids,
        min_object_size=MIN_OBJECT_SIZE,
        post_processed=POST_PROCESSING)
    test_pred_rle.extend(rle)
    test_pred_ids.extend(rle_id)

    if METHOD is None and (j % 500) == 0 and j:
        # Create submission file, save to disk and release some memory
        i += 1
        sub = pd.DataFrame()
        sub['ImageId'] = test_pred_ids
        sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(
            lambda x: ' '.join(str(y) for y in x))
        sub.to_csv('sub-dsbowl2018-{}.csv'.format(i), index=False)
        print('sub-dsbowl2018-{}.csv saved'.format(i))
        test_pred_rle = []
        test_pred_ids = []
        sub = []

# Create submission file
sub = pd.DataFrame()
sub['ImageId'] = test_pred_ids
sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(
    lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-{}.csv'.format(i + 1), index=False)
sub.head()

print('test_pred_ids.shape = {}'.format(np.array(test_pred_ids).shape))
print('test_pred_rle.shape = {}'.format(np.array(test_pred_rle).shape))

# Inspect a test prediction and check run length encoding.
# n = np.random.randint(len(x_test))
n = 171
pred = u_net.get_prediction_from_path(
    sess, [x_test[n]],
    method=METHOD,
    tgt_size=(IMG_HEIGHT, IMG_WIDTH),
    compatibility_multiplier=32,
    full_prediction=True)[0]
y_test_pred = utils.trsf_proba_to_binary(pred)
y_test_pred_original_size = utils.resize_as_original(
    [y_test_pred], [test_sizes[n]])[:, :, 0]
rle = list(utils.mask_to_rle(y_test_pred_original_size))
mask_rec = utils.rle_to_mask(rle, test_sizes[n])
print('Run length encoding: {} matches, {} misses'.format(
    np.sum((mask_rec == y_test_pred_original_size)),
    np.sum((mask_rec != y_test_pred_original_size))))

fig, axs = plt.subplots(2, 3, figsize=(20, 13))
axs[0, 0].imshow(utils.read_image(test_df['image_path'].loc[n]))
axs[0, 0].set_title('{}.) original test image'.format(n))
axs[0, 1].imshow(np.squeeze((utils.read_image(x_test[n]))))
axs[0, 1].set_title('{}.) transformed test image'.format(n))
axs[0, 2].imshow(np.squeeze(pred[:, :, 0]), cm.gray)
axs[0, 2].set_title('{}.) predicted test mask probabilities'.format(n))
axs[1, 0].imshow(np.squeeze(y_test_pred[:, :, 0]), cm.gray)
axs[1, 0].set_title('{}.) predicted test mask'.format(n))
axs[1, 1].imshow(np.squeeze(y_test_pred_original_size[:, :, 0]), cm.gray)
axs[1, 1].set_title(
    '{}.) predicted final test mask in original size'.format(n))
axs[1, 2].imshow(mask_rec, cm.gray)
axs[1, 2].set_title(
    '{}.) final mask recovered from run length encoding'.format(n))

sess.close()
