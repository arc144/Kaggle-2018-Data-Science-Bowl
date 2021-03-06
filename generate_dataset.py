import numpy as np
from skimage.morphology import watershed, binary_dilation
from scipy.ndimage import label
import matplotlib.pyplot as plt
import utils
import cv2
import os
from tqdm import tqdm

TRAIN_DIR = os.path.join(os.getcwd(), 'External datasets/external_data/train')
# TRAIN_DIR = os.path.join(os.getcwd(), 'stage1_train')
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks
IMG_TYPE = '.png'
train_df = utils.read_train_data_properties(TRAIN_DIR,
                                            IMG_DIR_NAME,
                                            MASK_DIR_NAME)

x_train, y_train = train_df['image_path'].values, train_df['mask_dir'].values

# Decide which run it is
run = 7
n = 7
chunk, rest = divmod(len(x_train), n)

x = x_train[run * chunk:(run + 1) * chunk]
y = y_train[run * chunk:(run + 1) * chunk]

# method = 'resize'
# tgt_size = (256, 256)

# x_loaded, y_loaded = utils.generate_images_masks(
#     x,
#     y,
#     method=method, tgt_size=tgt_size)

# for i in tqdm(range(len(x_loaded))):
#     np.save(x[i] + 'img.npy', x_loaded[i])
#     np.save(y[i] + 'mask2.npy', y_loaded[i])

x_loaded, y_loaded = utils.load_images_masks(x, y)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4)
axs[0].imshow(x_loaded[0])
axs[0].set_title('Original Image')
axs[1].imshow(1 - y_loaded[0][:, :, 0], cmap='Greys')
axs[1].set_title('Original Mask')
axs[2].imshow(1 - y_loaded[0][:, :, 2], cmap='Greys')
axs[2].set_title('Borderless Mask')
axs[3].imshow(1 - y_loaded[0][:, :, 3], cmap='Greys')
axs[3].set_title('Touching Borders')

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.show()
plt.tight_layout()
