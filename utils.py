import cv2
import os
import numpy as np
from scipy import ndimage
import pandas as pd
import sys
import tqdm
from scipy.ndimage import label
import keras.preprocessing.image
import skimage
from skimage.morphology import watershed, binary_dilation, binary_erosion
from data_augmentation import random_transform

# Collection of methods for data operations. Implemented are functions to read
# images/masks from files and to read basic properties of the train/test
# data sets.


def read_image(filepath, color_mode=cv2.IMREAD_COLOR,
               target_size=None, method='resize',
               seed=None):
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


def load_images_masks(x_paths, y_paths):
    '''Wrapper for load_images and load_masks'''
    imgs = []
    masks = []
    for im_path, mask_path in zip(x_paths, y_paths):
        imgs.append(np.load(im_path + 'img.npy'))
        masks.append(np.load(mask_path + 'mask2.npy'))

    return normalize(np.array(imgs)), normalize(np.array(masks))


def load_images(paths, tgt_size=None, color_mode=cv2.IMREAD_COLOR,
                method='resize'):
    '''Wrapper for loading images'''
    ret = []
    for path in paths:
        if method == 'resize':
            ret.append(read_image(path,
                                  color_mode=color_mode,
                                  target_size=tgt_size,
                                  method=method))

        elif method is None:
            ret.append(read_image(
                path,
                color_mode=color_mode,
                target_size=None,
                method=method))

    ret = np.array(ret)

    if len(ret.shape) == 3:
        ret = np.expand_dims(ret, axis=-1)

    ret = normalize(ret, type_=0)

    return ret


def load_masks(paths, tgt_size=None,
               method='resize'):
    '''Wrapper for loading masks'''
    ret = []
    for path in paths:
        if method == 'resize':
            ret.append(read_mask(path,
                                 target_size=tgt_size,
                                 method=method))

        elif method is None:
            ret.append(read_mask(
                path,
                target_size=None,
                method=method))

    ret = np.array(ret)

    if len(ret.shape) == 3:
        ret = np.expand_dims(ret, axis=-1)

    ret = normalize(ret, type_=0)

    return ret


def generate_images_masks(x_paths, y_paths, color_mode=cv2.IMREAD_COLOR,
                          tgt_size=None, method='resize',
                          seeds=None):
    '''Wrapper for load_images and load_masks'''
    if method == 'random_crop':
        imgs, seeds = generate_images(x_paths, color_mode=color_mode,
                                      method=method, tgt_size=tgt_size)
    else:
        imgs = generate_images(x_paths, color_mode=color_mode,
                               method=method, tgt_size=tgt_size)

    masks = generate_masks(y_paths,
                           method=method, tgt_size=tgt_size,
                           seeds=seeds)

    return imgs, masks


def generate_images(paths, color_mode=cv2.IMREAD_COLOR,
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
            ret.append(read_image(
                path,
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


def generate_masks(paths, tgt_size=None, method='resize', seeds=None):
    '''Wrapper for read_image multiple times'''
    ret = []

    for i, path in enumerate(paths):
        if method == 'resize':
            ret.append(generate_unjoint_mask(path,
                                             tgt_size=tgt_size))
        elif method is None:
            ret.append(generate_unjoint_mask(path,
                                             tgt_size=None))

    full_mask = normalize(np.array([x[0] for x in ret]), type_=0)
    borderless_mask = normalize(np.array([x[1] for x in ret]), type_=0)
    borders = normalize(np.array([x[2] for x in ret]), type_=0)

    if len(full_mask.shape) == 3:
        full_mask = np.expand_dims(full_mask, axis=-1)
        borderless_mask = np.expand_dims(borderless_mask, axis=-1)
        borders = np.expand_dims(borders, axis=-1)

    return np.concatenate(
        [full_mask, 1 - full_mask, borderless_mask, borders], -1)


def marker_from_mask(directory):
    '''Generate markers for watershed from masks'''
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, None)
        m = label(mask_tmp)[0]
        if not i:
            marker = (i + 1) * m
        else:
            marker = marker + (i + 1) * m
    return marker


def overlap_from_mask(directory, selem=3):
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, None)
        m = binary_dilation(
            label(mask_tmp)[0], selem=np.ones([selem, selem])).astype(np.int32)

        if not i:
            border = m
        else:
            border = np.add(border, m)
    border[border == 1] = 0
    border[border > 1] = 1
    return border.astype(np.uint8)


def overlap_and_markers_from_mask(directory, selem=4):
    '''Generate overlap borders and markers from mask'''
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, None)
        m = label(mask_tmp)[0]
        bm = binary_dilation(
            m, selem=np.ones([selem, selem])).astype(np.int32)

        if not i:
            border = bm
            marker = (i + 1) * m
        else:
            border = border + bm
            marker = marker + (i + 1) * m

    border[border == 1] = 0
    border[border > 1] = 1
    return border.astype(np.uint8), marker


def generate_unjoint_mask(directory, dilation_selem=5, erosion_selem=2, tgt_size=None):
    '''Generate mask, unjoint masks and borders to train'''
    mask = read_mask(directory)
    overlap, marker = overlap_and_markers_from_mask(directory, dilation_selem)
    unjoint_mask = watershed(mask, marker, mask=mask, watershed_line=True)
    unjoint_mask = binary_erosion(
        trsf_proba_to_binary(unjoint_mask),
        selem=np.ones([erosion_selem, erosion_selem])).astype(np.uint8)
    if tgt_size is not None:
        mask = cv2.resize(mask, tgt_size, interpolation=cv2.INTER_AREA)
        overlap = cv2.resize(overlap, tgt_size, interpolation=cv2.INTER_AREA)
        unjoint_mask = cv2.resize(
            unjoint_mask, tgt_size, interpolation=cv2.INTER_AREA)
    return mask, unjoint_mask * (1 - overlap), overlap

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
    if isinstance(y_data, np.ndarray):
        y_data = np.greater(y_data, threshold).astype(np.uint8)
    elif isinstance(y_data, list):
        for i in range(len(y_data)):
            y_data[i] = np.greater(y_data[i], threshold).astype(np.uint8)
    return y_data


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
            patches[h * n_crops_w + w, :, :, :] = im[start_h:end_h,
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
                             rotation_range=90,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.7,
                             zoom_range=0.3,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=True,
                             channel_shuffle=True,
                             seed=seed))
        ret_masks.append(
            random_transform(y, row_axis=0, col_axis=1, channel_axis=2,
                             rotation_range=90,
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
    return np.array(ret_imgs), np.array(trsf_proba_to_binary(ret_masks))


def match_size_with_pad(img, multiplier):
    '''Pad image with 0 if size is not multiple of multiplier
    Input: single ndarray image of shape HxWxC'''
    H, W = img.shape[0], img.shape[1]
    H_pad = H % multiplier
    W_pad = W % multiplier
    if H_pad or W_pad:
        hp = multiplier - H_pad
        wp = multiplier - W_pad
        img = np.pad(img, [[0, hp], [0, wp], [0, 0]], 'symmetric')
    else:
        hp, wp = 0, 0

    return img, (hp, wp)


def unpad_image_to_original_size(img, pad_size):
    '''Pad image with 0 if size is not multiple of multiplier
    Input: single ndarray image of shape HxWxC and tuple of ints'''
    if any(pad_size):
        return img[:-pad_size[0], :-pad_size[1], :]
    else:
        return img


def resize_as_original(y_test_pred, test_sizes):
    # Resize predicted masks to original image size.
    y_test_pred_original_size = []
    for i in range(len(y_test_pred)):
        original_size = test_sizes[i]
        if y_test_pred[i].shape[:2] != original_size:
            res_mask = trsf_proba_to_binary(
                skimage.transform.resize(
                    np.squeeze(y_test_pred[i]),
                    original_size,
                    mode='constant', preserve_range=True))

        else:
            res_mask = np.squeeze(y_test_pred[i])
        y_test_pred_original_size.append(res_mask)
    return np.array(y_test_pred_original_size)


def postprocessing(pred, method='watershed'):
    '''Apply postprocessing to predictions'''
    ret = []
    for i in range(len(pred)):
        full_mask = trsf_proba_to_binary(pred[i][:, :, 0])

        if method == 'watershed':
            borderless_mask = trsf_proba_to_binary(pred[i][:, :, 1])
            borders = trsf_proba_to_binary(pred[i][:, :, 2])
            markers = label(borderless_mask * (1 - borders))[0]
            ret.append(watershed(full_mask, markers=markers,
                                 mask=full_mask, watershed_line=False))
        else:
            ret.append(full_mask)
    return np.array(ret)


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


def mask_to_rle(mask, cutoff=.5, post_processed=False, min_object_size=20):
    """ Return run length encoding of mask. """
    # segment image and label different objects
    if not post_processed:
        lab_mask = skimage.morphology.label(mask > cutoff)
    else:
        lab_mask = mask

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


def mask_to_rle_wrapper(y_test_pred_original_size, test_ids,
                        post_processed=False, min_object_size=20):
    # Run length encoding of predicted test masks.
    test_pred_rle = []
    test_pred_ids = []
    for n, id_ in enumerate(test_ids):
        rle = list(mask_to_rle(
            y_test_pred_original_size[n],
            post_processed=post_processed,
            min_object_size=min_object_size))
        if len(rle) == 0:
            rle = [[]]
        test_pred_rle.extend(rle)
        test_pred_ids.extend([id_] * len(rle))
    return test_pred_rle, test_pred_ids


if __name__ == '__main__':
    x = np.random.randint(0, 100, size=[3, 3, 2])
    xp, pads = match_size_with_pad(x, 4)
    print(xp.shape, pads)
    print(xp)
    xup = unpad_image_to_original_size(xp, pads)
    print(xup.shape)
    print(xup)
