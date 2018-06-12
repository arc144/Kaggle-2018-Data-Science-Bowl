import matplotlib.pyplot as plt
import utils
import numpy as np
import skimage.morphology
import pandas as pd
import score
import os


def overview_trn_Img_n_Masks(train_df):
    # Overview of train images/masks. There is a lot of variation concerning
    # the form/size/number of nuclei and the darkness/lightness/colorfulness of
    # the images.
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    for i in range(4):
        n = np.random.randint(0, len(train_df))
        axs[i, 0].imshow(utils.read_image(train_df['image_path'].loc[n]))
        axs[i, 0].set_title('{}. image'.format(n))
        axs[i, 1].imshow(utils.read_mask(
            train_df['mask_dir'].loc[n]), cmap='gray')
        axs[i, 1].set_title('{}. mask'.format(n))
        axs[i, 2].imshow(utils.calculate_weights_from_dir(
            train_df['mask_dir'].loc[n]), cmap='jet')
        axs[i, 2].set_title('{}. weights'.format(n))


def get_nuclei_sizes(y_train):
    nuclei_sizes = []
    mask_idx = []
    for i in range(len(y_train)):
        mask = y_train[i].reshape(y_train.shape[1], y_train.shape[2])
        lab_mask = skimage.morphology.label(mask > .5)
        (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
        nuclei_sizes.extend(mask_sizes[1:])
        mask_idx.extend([i] * len(mask_sizes[1:]))
    return mask_idx, nuclei_sizes


def analyse_nuclei_sizes():
    # Analyze nuclei sizes.
    mask_idx, nuclei_sizes = get_nuclei_sizes()
    nuclei_sizes_df = pd.DataFrame()
    nuclei_sizes_df['mask_index'] = mask_idx
    nuclei_sizes_df['nucleous_size'] = nuclei_sizes

    print(nuclei_sizes_df.describe())
    nuclei_sizes_df.sort_values(by='nucleous_size', ascending=True).head(10)


def img_comparison_plot(train_df, x_train, y_train,  y_weights,
                        target_size, n):
    """Plot the original and transformed images/masks."""
    fig, axs = plt.subplots(1, 6, figsize=(20, 20))
    axs[0].imshow(utils.read_image(train_df['image_path'].loc[n]))
    axs[0].set_title('{}.) original image'.format(n))
    img, img_type = utils.imshow_args(x_train[n])
    axs[1].imshow(img, img_type)
    axs[1].set_title('{}.) transformed image'.format(n))
    axs[2].imshow(utils.read_mask(train_df['mask_dir'].loc[n]), cmap='jet')
    axs[2].set_title('{}.) original mask'.format(n))
    axs[3].imshow(y_train[n, :, :, 0], cmap='jet')
    axs[3].set_title('{}.) transformed mask'.format(n))
    axs[4].imshow(utils.calculate_weights(train_df['mask_dir'].loc[n],
                                          target_size=target_size), cmap='jet')
    axs[4].set_title('{}.) original weights'.format(n))
    axs[5].imshow(y_weights[n, :, :, 0], cmap='jet')
    axs[5].set_title('{}.) transformed weights'.format(n))


def plot_generated_image_mask(x_train, y_train, y_weights, n):
    # Generate new images/masks via transformations applied on the original
    # images/maks. Data augmentations can be used for regularization.
    fig, axs = plt.subplots(1, 6, figsize=(20, 20))
    img_new, mask_new, weights_new = utils.generate_images_and_masks(
        x_train[n:n + 1], y_train[n:n + 1], y_weights[n:n + 1])
    img, img_type = utils.imshow_args(x_train[n])
    axs[0].imshow(img, img_type)
    axs[0].set_title('{}. original image'.format(n))
    img, img_type = utils.imshow_args(img_new[0])
    axs[1].imshow(img, img_type)
    axs[1].set_title('{}. generated image'.format(n))
    axs[2].imshow(y_train[n, :, :, 0], cmap='gray')
    axs[2].set_title('{}. original mask'.format(n))
    axs[3].imshow(mask_new[0, :, :, 0], cmap='gray')
    axs[3].set_title('{}. generated mask'.format(n))
    axs[4].imshow(y_weights[n, :, :, 0], cmap='jet')
    axs[4].set_title('{}. weights'.format(n))
    axs[5].imshow(weights_new[0, :, :, 0], cmap='jet')
    axs[5].set_title('{}. generated weights'.format(n))


def check_score_metric(n, train_df, y_train):
    # Check the score metric for one sample. The predicted mask is simulated
    # and can be modified in order to check the correct implementation of
    # the score metric.
    true_mask = y_train[n, :, :, 0].copy()
    lab_true_mask = score.get_labeled_mask(true_mask)
    pred_mask = true_mask.copy()  # Create predicted mask from true mask.
    true_mask[lab_true_mask == 7] = 0  # Remove one object => false postive
    pred_mask[lab_true_mask == 10] = 0  # Remove one object => false negative
    offset = 5  # Offset.
    pred_mask = pred_mask[offset:, offset:]
    pred_mask = np.pad(pred_mask, ((0, offset), (0, offset)), mode="constant")
    score.plot_score_summary(n, train_df, true_mask, pred_mask)


def check_num_identifiable_obj(y_train):
    # Study how many objects in the masks can be identified.
    # This is a limiting factor for the overall performance.
    min_pixels_per_object = 20
    summary = []
    for n in range(len(y_train)):
        img = y_train[n, :, :, 0]
        lab_img = score.get_labeled_mask(img)
        img_labels, img_area = np.unique(lab_img, return_counts=True)
        img_labels = img_labels[img_area >= min_pixels_per_object]
        img_area = img_area[img_area >= min_pixels_per_object]
        n_true_labels = train_df['num_masks'][n]
        n_ident_labels = len(img_labels)
        diff = np.abs(n_ident_labels - n_true_labels)
        summary.append([n_true_labels, n_ident_labels, diff])

    sum_df = pd.DataFrame(summary, columns=(
        ['true_objects', 'identified_objects', 'subtraction']))
    sum_df.describe()


if __name__ == '__main__':
    # Global variables.
    min_object_size = 20       # Minimal nucleous size in pixels
    x_train = []
    y_train = []
    x_test = []
    # Dirs
    CW_DIR = os.getcwd()
    TRAIN_DIR = os.path.join(CW_DIR, 'stage1_train')
    TEST_DIR = os.path.join(CW_DIR, 'stage1_final_test')
    IMG_TYPE = '.png'         # Image type
    IMG_DIR_NAME = 'images'   # Folder name including the image
    MASK_DIR_NAME = 'masks'   # Folder name including the masks
    LOGS_DIR_NAME = 'logs'    # Folder name for TensorBoard summaries
    # Display working/train/test directories.
    print('CW_DIR = {}'.format(CW_DIR))
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

    # Read images/masks from files and resize them. Each image and mask
    # is stored as a 3-dim array where the number of channels is 3 and 1,
    # respectively.
    x_train, y_train, y_weights = utils.load_raw_data(train_df)
