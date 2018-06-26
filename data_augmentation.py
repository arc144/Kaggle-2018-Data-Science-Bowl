import cv2
import numpy as np
from scipy import linalg
import scipy.ndimage as ndi
from imgaug import augmenters as iaa
import imgaug as ia
import utils


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(
            transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(
            transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
            transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def apply_channel_shift(x, intensity, channel_axis=0):
    """Performs a channel shift.
    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.
    # Returns
        Numpy image tensor.
    """
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [
        np.clip(x_channel + intensity,
                min_x,
                max_x)
        for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def apply_channel_shuffle(x, channel_axis=2):
    '''Shuffle image channels'''
    x = np.rollaxis(x, channel_axis, 0)
    rand = np.arange(x.shape[0])
    np.random.shuffle(rand)
    x = x[rand, :, :]
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_rotation(x, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.random.uniform(-rg, rg)
    x = apply_affine_transform(x, theta=theta, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_shift(x, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_shear(x, intensity, row_axis=0, col_axis=1, channel_axis=2,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    x = apply_affine_transform(x, shear=shear, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_zoom(x, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = apply_affine_transform(x, zx=zx, zy=zy, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_channel_shift(x, intensity_range, channel_axis=0):
    """Performs a random channel shift.
    # Arguments
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.
    # Returns
        Numpy image tensor.
    """
    intensity = np.random.uniform(-intensity_range, intensity_range)
    return apply_channel_shift(x, intensity, channel_axis=channel_axis)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def get_random_transform(img_shape, params, row_axis=0, col_axis=1, seed=None):
    """Generates random parameters for a transformation.
    # Arguments
        seed: Random seed.
        img_shape: Tuple of integers. Shape of the image that is transformed.
    # Returns
        A dictionary containing randomly chosen parameters describing the
        transformation.
    """
    rotation_range = params['rotation_range']
    height_shift_range = params['height_shift_range']
    width_shift_range = params['width_shift_range']
    width_shift_range = params['width_shift_range']
    shear_range = params['shear_range']
    horizontal_flip = params['horizontal_flip']
    vertical_flip = params['vertical_flip']
    channel_shift_range = params['channel_shift_range']
    zoom_range = params['zoom_range']
    channel_shuffle = params['channel_shuffle']

    if seed is not None:
        np.random.seed(seed)

    if rotation_range:
        theta = np.random.uniform(
            -rotation_range,
            rotation_range)
    else:
        theta = 0

    if height_shift_range:
        try:  # 1-D array-like or int
            tx = np.random.choice(height_shift_range)
            tx *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            tx = np.random.uniform(-height_shift_range,
                                   height_shift_range)
        if np.max(height_shift_range) < 1:
            tx *= img_shape[row_axis]
    else:
        tx = 0

    if width_shift_range:
        try:  # 1-D array-like or int
            ty = np.random.choice(width_shift_range)
            ty *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            ty = np.random.uniform(-width_shift_range,
                                   width_shift_range)
        if np.max(width_shift_range) < 1:
            ty *= img_shape[col_axis]
    else:
        ty = 0

    if shear_range:
        shear = np.random.uniform(
            -shear_range,
            shear_range)
    else:
        shear = 0

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(
            zoom_range[0],
            zoom_range[1],
            2)

    flip_horizontal = (np.random.random() < 0.5) * horizontal_flip
    flip_vertical = (np.random.random() < 0.5) * vertical_flip
    channel_shuffle = (np.random.random() < 0.5) * channel_shuffle

    channel_shift_intensity = None
    if channel_shift_range != 0:
        channel_shift_intensity = np.random.uniform(-channel_shift_range,
                                                    channel_shift_range)

    transform_parameters = {'theta': theta,
                            'tx': tx,
                            'ty': ty,
                            'shear': shear,
                            'zx': zx,
                            'zy': zy,
                            'flip_horizontal': flip_horizontal,
                            'flip_vertical': flip_vertical,
                            'channel_shuffle': channel_shuffle,
                            'channel_shift_intensity': channel_shift_intensity}

    return transform_parameters


def apply_transform(x, transform_parameters, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    """Applies a transformation to an image according to given parameters.
    # Arguments
        x: 3D tensor, single image.
        transform_parameters: Dictionary with string - parameter pairs
            describing the transformation. Currently, the following parameters
            from the dictionary are used:
            - `'theta'`: Float. Rotation angle in degrees.
            - `'tx'`: Float. Shift in the x direction.
            - `'ty'`: Float. Shift in the y direction.
            - `'shear'`: Float. Shear angle in degrees.
            - `'zx'`: Float. Zoom in the x direction.
            - `'zy'`: Float. Zoom in the y direction.
            - `'flip_horizontal'`: Boolean. Horizontal flip.
            - `'flip_vertical'`: Boolean. Vertical flip.
            - `'channel_shift_intencity'`: Float. Channel shift intensity.
            - `'brightness'`: Float. Brightness shift intensity.
    # Returns
        A ransformed version of the input (same shape).
    """
    # x is a single image, so it doesn't have image number at index 0

    x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                               transform_parameters.get('tx', 0),
                               transform_parameters.get('ty', 0),
                               transform_parameters.get('shear', 0),
                               transform_parameters.get('zx', 1),
                               transform_parameters.get('zy', 1),
                               row_axis=row_axis, col_axis=col_axis,
                               channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)

    if transform_parameters.get('channel_shift_intensity') is not None:
        x = apply_channel_shift(x,
                                transform_parameters[
                                    'channel_shift_intensity'],
                                channel_axis)

    if transform_parameters.get('flip_horizontal', False):
        x = flip_axis(x, col_axis)

    if transform_parameters.get('flip_vertical', False):
        x = flip_axis(x, row_axis)

    if transform_parameters.get('channel_shuffle', False):
        x = apply_channel_shuffle(x)
    return x


def random_transform(x, row_axis=0, col_axis=1, channel_axis=2,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     channel_shuffle=False,
                     seed=None):
    """Applies a random transformation to an image.
    # Arguments
        x: 3D tensor, single image.
        seed: Random seed.
    # Returns
        A randomly transformed version of the input (same shape).
    """
    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = [zoom_range[0], zoom_range[1]]
    else:
        raise ValueError('`zoom_range` should be a float or '
                         'a tuple or list of two floats. '
                         'Received: %s' % zoom_range)

    params = dict()
    params['rotation_range'] = rotation_range
    params['width_shift_range'] = width_shift_range
    params['height_shift_range'] = height_shift_range
    params['shear_range'] = shear_range
    params['zoom_range'] = zoom_range
    params['channel_shift_range'] = channel_shift_range
    params['fill_mode'] = fill_mode
    params['cval'] = cval
    params['horizontal_flip'] = horizontal_flip
    params['vertical_flip'] = vertical_flip
    params['channel_shuffle'] = channel_shuffle

    transform_params = get_random_transform(
        x.shape, params, row_axis=row_axis, col_axis=col_axis, seed=seed)
    return apply_transform(x, transform_params)


def transform(x, row_axis=0, col_axis=1, channel_axis=2,
              rotation=0.,
              width_shift=0.,
              height_shift=0.,
              shear=0.,
              zoom=0.,
              channel_shift=0.,
              fill_mode='nearest',
              cval=0.,
              horizontal_flip=False,
              vertical_flip=False,
              seed=None):
    """Applies a random transformation to an image.
    # Arguments
        x: 3D tensor, single image.
        seed: Random seed.
    # Returns
        A randomly transformed version of the input (same shape).
    """
    if np.isscalar(zoom):
        zoom = [1 - zoom, 1 + zoom]
    elif len(zoom) == 2:
        zoom = [zoom[0], zoom[1]]
    else:
        raise ValueError('`zoom` should be a float or '
                         'a tuple or list of two floats. '
                         'Received: %s' % zoom)

    transform_parameters = {'theta': rotation,
                            'tx': width_shift,
                            'ty': height_shift,
                            'shear': shear,
                            'zx': zoom[0],
                            'zy': zoom[1],
                            'flip_horizontal': horizontal_flip,
                            'flip_vertical': vertical_flip,
                            'channel_shift_intensity': channel_shift}

    return apply_transform(x, transform_parameters)


# ####################### IMGAUG IMPLEMENTATION ########################
class Augmentation():

    def __init__(self):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every
        # second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),   # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-30, 30),  # shear by -30 to +30 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the
                    # top for examples)
                    mode=ia.ALL
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too
                # strong
                iaa.SomeOf((2, 5),
                           [
                    iaa.OneOf([
                        # blur images with a sigma between 0 and 3.0
                        iaa.GaussianBlur((0, 3.0)),
                        # blur image using local means with kernel sizes between 2
                        # and 7
                        iaa.AverageBlur(k=(2, 7)),
                        # blur image using local medians with kernel sizes between
                        # 2 and 7
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(
                        0.75, 1.5)),  # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(
                        0, 2.0)),  # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(
                        0.0, 0.02 * 255), per_channel=0.5),

                    # randomly remove up to 5% of the pixels
                    iaa.Dropout((0.01, 0.05), per_channel=False),

                    # change brightness of images (by -20 to 20 of original
                    # value)
                    iaa.Add((-20, 20), per_channel=0.5),
                    # change hue and saturation
                    iaa.AddToHueAndSaturation((-20, 20)),
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    # improve or worsen the contrast
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    # sometimes move parts of the image around
                    # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                    random_order=True
                )
            ],
            random_order=True
        )

        # change the activated augmenters for masks
        def activator_masks(images, augmenter, parents, default):
            if augmenter.name in ["UnnamedSomeOf"]:
                return False
            else:
                # default value for all other augmenters
                return default

        self.hooks_masks = ia.HooksImages(activator=activator_masks)

    def augment_img_and_mask(self, img, mask):
        seq_det = self.seq.to_deterministic()
        images_aug = seq_det.augment_images(img.astype(np.uint8))
        mask_aug = seq_det.augment_images(
            mask.astype(np.uint8), hooks=self.hooks_masks)
        return utils.normalize(images_aug), utils.normalize(mask_aug)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    TRAIN_PATH = 'test_image.png'
    im = plt.imread(TRAIN_PATH)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im)
    print(im.shape)
    # test aug
    alt = random_transform(im, row_axis=0, col_axis=1, channel_axis=2,
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
                           seed=None)
    axs[1].imshow(alt)
    plt.show()
