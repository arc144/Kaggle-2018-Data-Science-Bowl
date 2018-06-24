from utils import read_image, read_mask, normalize
from skimage.morphology import binary_dilation
import cv2
import numpy as np


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
                    tgt_size=None, method='resize',
                    check_compatibility=False,
                    compatibility_multiplier=32, seed=None):
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
                check_compatibility=check_compatibility,
                compatibility_multiplier=compatibility_multiplier,
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


def generate_unjoint_mask(directory, selem=3, tgt_size=None):
    '''Generate unjoint masks to train'''
    mask = read_mask(directory)
    overlap = overlap_from_mask(directory, selem)
    if tgt_size is not None:
        mask = cv2.resize(mask, tgt_size, interpolation=cv2.INTER_AREA)
        overlap = cv2.resize(overlap, tgt_size, interpolation=cv2.INTER_AREA)
    return mask, mask * (1 - overlap), overlap
