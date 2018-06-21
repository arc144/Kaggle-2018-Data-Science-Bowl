import cv2
import os
import numpy as np
from scipy import ndimage
import pandas as pd
import sys
import tqdm
import keras.preprocessing.image
import skimage
from data_augmentation import random_transform

# Collection of methods for data operations. Implemented are functions to read
# images/masks from files and to read basic properties of the train/test
# data sets.


def read_image(filepath, color_mode=cv2.IMREAD_COLOR,
               target_size=None, method='resize', seed=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        if method == 'resize':
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        elif method == 'crop':
            img = patchfy(img, crop_size=target_size)
        elif method == 'random_crop':
            img, seed = random_crop(
                img, crop_size=target_size, seed=seed, return_seed=True)
            return img, seed

    return img


def read_mask(directory, target_size=None, method='resize', seed=None):
    """Read and resize masks contained in a given directory."""
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, None)

        if not i:
            mask = mask_tmp
        else:
            mask = np.maximum(mask, mask_tmp)

    if target_size:
        if method == 'resize':
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
        elif method == 'crop':
            mask = patchfy(mask, crop_size=target_size)
        elif method == 'random_crop':
            mask = random_crop(mask, crop_size=target_size, seed=seed)
    return mask


def calculate_weights_from_dir(directory, target_size=None):
    """Read and resize masks contained in a given directory."""
    list_of_masks = []
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, None)
        if target_size:
            mask_tmp = cv2.resize(mask_tmp, target_size,
                                  interpolation=cv2.INTER_AREA)
        list_of_masks.append(mask_tmp)
        if not i:
            merged_mask = mask_tmp
        else:
            merged_mask = np.maximum(merged_mask, mask_tmp)

    weights = calculate_weight(
        merged_mask, list_of_masks)  # list is grey

    return weights


def calculate_weight(merged_mask, masks, w0=10, q=5):
    weight = np.zeros(merged_mask.shape)
    # calculate weight for important pixels
    distances = np.array(
        [ndimage.distance_transform_edt(m == 0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    weight = w0 * np.exp(-(d1 + d2)**2 / (2 * q**2)).astype(np.float32)
    weight = (merged_mask == 0) * weight
    return weight


def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(train_dir))[1]):

        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name)
        num_masks = len(next(os.walk(mask_dir))[2])
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], num_masks,
                    img_path, mask_dir])

    train_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                          'img_ratio', 'num_channels',
                                          'num_masks', 'image_path',
                                          'mask_dir'])
    return train_df


def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(test_dir))[1]):

        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                         'img_ratio', 'num_channels',
                                         'image_path'])
    return test_df


def imshow_args(x):
    """Matplotlib imshow arguments for plotting."""
    if len(x.shape) == 2:
        return x
    if x.shape[2] == 1:
        return x[:, :, 0]
    elif x.shape[2] == 3:
        return x


def load_raw_data(train_df, image_size=(256, 256), method='resize'):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_train, y_train, y_weights = [], [], []

    # Read and resize train images/masks.
    print('Loading and resizing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']),
                                 total=len(train_df)):
        img = read_image(
            train_df['image_path'].loc[i],
            target_size=image_size,
            method=method)
        mask = read_mask(
            train_df['mask_dir'].loc[i],
            target_size=image_size,
            method=method)
        weights = calculate_weights_from_dir(train_df['mask_dir'].loc[i],
                                             target_size=image_size)
        if method == 'resize':
            x_train.append(img)
            y_train.append(mask)
        elif method == 'crop':
            x_train.extend(img)
            y_train.extend(mask)

        y_weights.append(weights)

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    y_train = np.expand_dims(np.array(y_train), axis=3)
    y_weights = np.expand_dims(np.array(y_weights), axis=3)

    print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train.shape: {} of dtype {}'.format(y_train.shape, x_train.dtype))
    print('y_weights.shape: {} of dtype {}'.format(
        y_weights.shape, y_weights.dtype))

    return x_train, y_train, y_weights


def load_test_data(test_df, image_size=(256, 256)):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_test = []

    # Read and resize test images.
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']),
                                 total=len(test_df)):
        img = read_image(test_df['image_path'].loc[i], target_size=image_size)
        x_test.append(img)

    x_test = np.array(x_test)

    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))

    return x_test


def load_images_masks(x_paths, y_paths, color_mode=cv2.IMREAD_COLOR,
                      tgt_size=None, method='resize', seeds=None):
    '''Wrapper for load_images and load_masks'''
    if method == 'random_crop':
        imgs, seeds = load_images(x_paths, color_mode=color_mode,
                                  method=method, tgt_size=tgt_size)
    else:
        imgs = load_images(x_paths, color_mode=color_mode,
                           method=method, tgt_size=tgt_size)

    masks = load_masks(y_paths,
                       method=method, tgt_size=tgt_size,
                       seeds=seeds)
    return imgs, masks


def load_images(paths, color_mode=cv2.IMREAD_COLOR,
                tgt_size=None, method='resize', seed=None):
    '''Wrapper for read_image multiple times'''
    ret = []
    seeds = []
    for path in paths:
        if method == 'crop':
            ret.extend(read_image(path,
                                  color_mode=color_mode,
                                  target_size=tgt_size,
                                  method=method,
                                  seed=seed))
        elif method == 'random_crop':
            im, seed = read_image(path,
                                  color_mode=color_mode,
                                  target_size=tgt_size,
                                  method=method,
                                  seed=seed)
            ret.append(im)
            seeds.append(seed)

        elif method == 'resize':
            ret.append(read_image(path,
                                  color_mode=color_mode,
                                  target_size=tgt_size,
                                  method=method,
                                  seed=seed))

        elif method is None:
            ret.append(read_image(path,
                                  color_mode=color_mode,
                                  target_size=None,
                                  method=method,
                                  seed=seed))

    ret = np.array(ret)

    if len(ret.shape) == 3:
        ret = np.expand_dims(ret, axis=-1)

    ret = normalize(ret, type_=0)

    if method == 'random_crop':
        return ret, seeds
    else:
        return ret


def load_masks(paths, tgt_size=None, method='resize', seeds=None):
    '''Wrapper for read_image multiple times'''
    ret = []

    for i, path in enumerate(paths):
        if method == 'crop':
            ret.extend(read_mask(path,
                                 target_size=tgt_size,
                                 method=method))
        elif method == 'random_crop':
            ret.append(read_mask(path,
                                 target_size=tgt_size,
                                 method=method,
                                 seed=seeds[i]))

        elif method == 'resize':
            ret.append(read_mask(path,
                                 target_size=tgt_size,
                                 method=method))
        elif method is None:
            ret.append(read_mask(path,
                                 target_size=None,
                                 method=method))

    ret = np.array(ret)
    if len(ret.shape) == 3:
        ret = np.expand_dims(ret, axis=-1)
    return normalize(ret, type_=0)

# Collection of methods for basic data manipulation like normalizing,
# inverting, color transformation and generating new images/masks


def normalize_imgs(data):
    """Normalize images."""
    return normalize(data, type_=1)


def normalize_masks(data):
    """Normalize masks."""
    return normalize(data, type_=1)


def normalize(data, type_=0):
    """Normalize data."""
    if len(data.shape) >= 3:
        if type_ == 0:
            # Convert pixel values from [0:255] to [0:1] by global factor
            data = (data - data.min()) / (data.max() - data.min())
        if type_ == 1:
            # Convert pixel values from [0:255] to [0:1] by local factor
            div = data.max(axis=tuple(
                np.arange(1, len(data.shape))), keepdims=True)
            div[div < 0.01 * data.mean()] = 1.  # protect against too small pixel intensities
            data = data.astype(np.float32) / div
        if type_ == 2:
            # Standardisation of each image
            data = data.astype(np.float32) / data.max()
            mean = data.mean(axis=tuple(
                np.arange(1, len(data.shape))), keepdims=True)
            std = data.std(axis=tuple(
                np.arange(1, len(data.shape))), keepdims=True)
            data = (data - mean) / std

    # One by one in case images have different shapes
    elif len(data.shape) == 1:
        data = list(data)
        for i in range(len(data)):
            data[i] = (data[i] - data[i].min()) / \
                (data[i].max() - data[i].min())

    return np.array(data)


def trsf_proba_to_binary(y_data, threshold=0.5):
    """Transform propabilities into binary values 0 or 1."""
    return np.greater(y_data, threshold).astype(np.uint8)


def invert_imgs(imgs, cutoff=.5):
    '''Invert image if mean value is greater than cutoff.'''
    imgs = np.array(
        list(map(lambda x: 1. - x if np.mean(x) > cutoff else x, imgs)))
    return normalize_imgs(imgs)


def imgs_to_grayscale(imgs):
    '''Transform RGB images into grayscale spectrum.'''
    if imgs.shape[3] == 3:
        imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
    return imgs


def generate_images(imgs, seed=None):
    """Generate new images."""
    # Transformations.
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90., width_shift_range=0.02, height_shift_range=0.02,
        zoom_range=0.10, horizontal_flip=True, vertical_flip=True)

    # Generate new set of images
    imgs = image_generator.flow(imgs, np.zeros(len(imgs)),
                                batch_size=len(imgs),
                                shuffle=False, seed=seed).next()
    return imgs[0]


def generate_images_and_masks(imgs, masks, weights=None):
    """Generate new images and masks."""
    seed = np.random.randint(10000)
    imgs = generate_images(imgs, seed=seed)
    if weights is not None:
        weights = generate_images(weights, seed=seed)
    masks = trsf_proba_to_binary(generate_images(masks, seed=seed))
    return imgs, masks, weights


def preprocess_raw_data(x_train, y_train, y_weights,
                        grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_train = normalize_imgs(x_train)
    y_train = trsf_proba_to_binary(normalize_masks(y_train))
    y_weights = normalize(y_weights, type_=0)
    print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_train = imgs_to_grayscale(x_train)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_train = invert_imgs(x_train)
        print('Images inverted to remove light backgrounds.')

    return x_train, y_train, y_weights


def preprocess_test_data(x_test, grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_test = normalize_imgs(x_test)
    print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_test = imgs_to_grayscale(x_test)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_test = invert_imgs(x_test)
        print('Images inverted to remove light backgrounds.')

    return x_test


def patchfy(im, crop_size=(256, 256)):
    '''Split an image into crops. If dimensions does not
    match, the last crops are overlapped with the ones before last.'''

    # Convert image into np.array and verify shape
    im = np.array(im)
    if len(im.shape) == 3:
        height, width, channels = im.shape
    elif len(im.shape) == 2:
        height, width = im.shape
        channels = 1
        im = np.expand_dims(im, axis=-1)
    else:
        raise IndexError('Expecting an image of shape (H,W,C)' +
                         ' or (H, W), instead got {}'.format(im.shape))
    # Verify if crop_size is list or tuple, otherwise assume square crops
    if isinstance(crop_size, int) or isinstance(crop_size, float):
        crop_size = (crop_size, crop_size)

    # Number of total crops in each dimension
    # last crop will have overlap to match dimensions
    quotient_h, rest_h = divmod(height, crop_size[0])
    quotient_w, rest_w = divmod(width, crop_size[1])

    n_crops_h = quotient_h if not rest_h else quotient_h + 1
    n_crops_w = quotient_w if not rest_w else quotient_w + 1
    # Create empty array that will receive the crops
    patches = np.empty(shape=(n_crops_h * n_crops_w,
                              crop_size[0],
                              crop_size[1],
                              channels))

    # Loop for height
    for h in range(n_crops_h):
        # Loop for width
        for w in range(n_crops_w):
            start_h = h * crop_size[0]
            start_w = w * crop_size[1]
            end_h = start_h + crop_size[0]
            end_w = start_w + crop_size[1]
            # Deal with last patches
            if end_h > height:
                start_h = height - crop_size[0]
                end_h = height
            if end_w > width:
                start_w = width - crop_size[1]
                end_w = width
            # Add crop to patches array
            patches[h * n_crops_w + w, :, :, :] = \
                im[start_h:end_h,
                   start_w:end_w,
                    :]
    return patches


def random_crop(im, crop_size=(256, 256), seed=None, return_seed=False, threshold=0.33):
    '''Randomly crop a image'''
    height, width = im.shape[0:2]
    if not seed:
        x = np.random.random_integers(0, height - crop_size[0])
        y = np.random.random_integers(0, width - crop_size[1])
    else:
        x, y = seed
    if len(im.shape) < 3:
        ret = im[x:x + crop_size[0], y:y + crop_size[1]]
    elif len(im.shape) == 3:
        ret = im[x:x + crop_size[0], y:y + crop_size[1], :]

    # Check unbalancing
    # if np.sum(ret) / (crop_size[0] * crop_size[1]) < threshold:
    #     return random_crop(im, crop_size=crop_size, seed=seed,
    #                        return_seed=return_seed, threshold=threshold)

    if return_seed:
        return ret, seed
    return ret


def augment_images_masks(imgs, masks):
    '''Augment images and masks, expects 4d arrays'''
    ret_imgs = []
    ret_masks = []
    for x, y in zip(imgs, masks):
        seed = np.random.randint(10000)
        ret_imgs.append(
            random_transform(x, row_axis=0, col_axis=1, channel_axis=2,
                             rotation_range=360,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.7,
                             zoom_range=0.3,
                             channel_shift_range=0.5,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=True,
                             channel_shuffle=True,
                             seed=seed))
        ret_masks.append(
            random_transform(y, row_axis=0, col_axis=1, channel_axis=2,
                             rotation_range=360,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.7,
                             zoom_range=0.3,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=True,
                             channel_shuffle=False,
                             seed=seed))
    return np.array(ret_imgs), np.array(ret_masks)


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
