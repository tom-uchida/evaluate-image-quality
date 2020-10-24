#
# リピートレベルごとの平均輝度値を出力するプログラム
# 背景の輝度値は平均計算から除外
#

import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

# Check arguments
import sys
args = sys.argv
if len(args) != 3:
    print("\nUSAGE : $ python calc_mean_pixel_value.py [original_image.bmp] [noise_image.bmp]")
    sys.exit()

# Figure setting
fig, ax = plt.subplots(3, figsize=(9, 8)) # figsize(width, height)
fig.subplots_adjust(hspace=0.4, wspace=0.4) # interval
ax[0] = plt.subplot2grid((2,2), (0,0))
ax[1] = plt.subplot2grid((2,2), (0,1))
ax[2] = plt.subplot2grid((2,2), (1,0), colspan=2)

# Read input image
def read_img(_img_name):
    # read input image
    img = cv2.imread(_img_name)

    # convert color (BGR → RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# Read two images
img_original_RGB    = read_img( args[1] )
img_noise_RGB       = read_img( args[2] )

# Convert RGB to Gray
img_original_Gray   = cv2.cvtColor(img_original_RGB, cv2.COLOR_RGB2GRAY)
img_noise_Gray      = cv2.cvtColor(img_noise_RGB,    cv2.COLOR_RGB2GRAY)

# Show input image
def show_img(_i, _img, _img_name):
    ax[_i].set_title(_img_name)

    # show image
    ax[_i].imshow(_img)

    return

#show_img(0, img_origin_RL1,    "Original Image (RL=1)")

# Get statistical data of pixel value
def get_data_of_pixel_value(_img, _img_name):
    print("===== Statistical Data of", _img_name, "(Gray) =====")
    print("Num of pixel values (== 255) :", np.sum(_img == 255))
    print("Num of pixel values (<= 1)   :", np.sum(_img <= 1))
    print("Num of pixel values (== 0)   :", np.sum(_img == 0) )
    print("Max                          :", np.max(_img))
    print("Min                          :", np.min(_img))
    #print("\nAverage :", np.mean(_img))
    #print("Median  :", np.median(_img))
    print("Average pixel value          :", _img[_img != 0].mean())
    # print("Standard Deviation           :", _img[_img != 0].std())
    print("\n")

    return _img[_img != 0].mean()

get_data_of_pixel_value(img_original_Gray,  "Original Image")
get_data_of_pixel_value(img_noise_Gray,     "Noise Image")

# Figure
# ----- originl image data -----
#ax[2].hist(gray_img_origin_RL1.ravel(), bins=50, color='red', alpha=0.5, label="Original Image (RL=1)")
#ax[2].hist(gray_img_origin_RL100.ravel(), bins=50, color='red', alpha=0.5, label="Original Image (RL=100)")

# ----- noised image data -----
# ax[2].hist(img_noised_RL1[:, :, 0].ravel(), bins=50, color='red', alpha=0.5, label="Noised Image (RL=1)")
# ax[2].hist(img_noised_RL100[:, :, 0].ravel(), bins=50, color='blue', alpha=0.5, label="Noised Image (RL=100)")
# ax[2].axvline(R_mean_noised_RL1, color='red')
# ax[2].axvline(R_mean_noised_RL100, color='blue')

# ax[2].set_title("Histogram of Pixel Value of Gray Scale", fontsize=12)
# ax[2].set_xlabel("Pixel value", fontsize=12)
# ax[2].set_ylabel("Number of pixels", fontsize=12)
# ax[2].set_xlim([10, 266])
#ax[2].set_ylim([0, 250000])
#plt.grid()
# ax[2].legend(fontsize=12)

# fig.show()
# plt.show()