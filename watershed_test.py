import utils
import matplotlib.pyplot as plt

im_path = '/media/data/Kaggle/Kaggle-2018-Data-Science-Bowl/stage1_train/709e094e39629a9ca21e187f007b331074694e443db40289447c1111f7e267e7/images/709e094e39629a9ca21e187f007b331074694e443db40289447c1111f7e267e7.png'
mask_path = '/media/data/Kaggle/Kaggle-2018-Data-Science-Bowl/stage1_train/709e094e39629a9ca21e187f007b331074694e443db40289447c1111f7e267e7/masks/'

im = utils.read_image(im_path)
mask = utils.read_mask(mask_path)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(im)
axs[1].imshow(mask, cmap='jet')
plt.show()
