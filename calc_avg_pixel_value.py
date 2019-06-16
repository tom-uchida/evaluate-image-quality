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



# ------------------------------
# ----- Placement settings -----
# ------------------------------
fig, ax = plt.subplots(3, figsize=(9, 8)) # figsize(width, height)
fig.subplots_adjust(hspace=0.4, wspace=0.4) # interval
ax[0] = plt.subplot2grid((2,2), (0,0))
ax[1] = plt.subplot2grid((2,2), (0,1))
ax[2] = plt.subplot2grid((2,2), (1,0), colspan=2)



# ----------------------------
# ----- Read input image -----
# ----------------------------
def read_img(_img_name):
  # read input image
  img = cv2.imread(_img_name)

  # convert color (BGR → RGB)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img

img_origin_RL1   = read_img('images/DATA/uniformly/1024/plane_10M_RL1.bmp')
img_origin_RL5   = read_img('images/DATA/uniformly/1024/plane_10M_RL1.bmp')
img_origin_RL10  = read_img('images/DATA/uniformly/1024/plane_10M_RL10.bmp')
img_origin_RL20  = read_img('images/DATA/uniformly/1024/plane_10M_RL20.bmp')
img_origin_RL30  = read_img('images/DATA/uniformly/1024/plane_10M_RL30.bmp')
img_origin_RL40  = read_img('images/DATA/uniformly/1024/plane_10M_RL40.bmp')
img_origin_RL50  = read_img('images/DATA/uniformly/1024/plane_10M_RL50.bmp')
img_origin_RL60  = read_img('images/DATA/uniformly/1024/plane_10M_RL60.bmp')
img_origin_RL70  = read_img('images/DATA/uniformly/1024/plane_10M_RL70.bmp')
img_origin_RL80  = read_img('images/DATA/uniformly/1024/plane_10M_RL80.bmp')
img_origin_RL90  = read_img('images/DATA/uniformly/1024/plane_10M_RL90.bmp')
img_origin_RL100 = read_img('images/DATA/uniformly/1024/plane_10M_RL100.bmp')
img_origin_RL150 = read_img('images/DATA/uniformly/1024/plane_10M_RL150.bmp')
img_origin_RL200 = read_img('images/DATA/uniformly/1024/plane_10M_RL200.bmp')
img_origin_RL300 = read_img('images/DATA/uniformly/1024/plane_10M_RL300.bmp')
img_origin_RL400 = read_img('images/DATA/uniformly/1024/plane_10M_RL400.bmp')
img_origin_RL500 = read_img('images/DATA/uniformly/1024/plane_10M_RL500.bmp')

img_noised_RL1   = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL1.bmp')
img_noised_RL5   = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL5.bmp')
img_noised_RL10  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL10.bmp')
img_noised_RL20  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL20.bmp')
img_noised_RL30  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL30.bmp')
img_noised_RL40  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL40.bmp')
img_noised_RL50  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL50.bmp')
img_noised_RL60  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL60.bmp')
img_noised_RL70  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL70.bmp')
img_noised_RL80  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL80.bmp')
img_noised_RL90  = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL90.bmp')
img_noised_RL100 = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL100.bmp')
img_noised_RL150 = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL150.bmp')
img_noised_RL200 = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL200.bmp')
img_noised_RL300 = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL300.bmp')
img_noised_RL400 = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL400.bmp')
img_noised_RL500 = read_img('images/DATA/uniformly/1024/gaussian_9M_1M_RL500.bmp')
# image information（height × width × 色数）
# print("img_origin : ", img_origin.shape)  
# print("img_noised : ", img_noised.shape)
# print("\n")



# ----------------------------
# ----- Show input image -----
# ----------------------------
def show_img(_i, _img, _img_name):
  ax[_i].set_title(_img_name)

  # show image
  ax[_i].imshow(_img)

  return

#show_img(0, img_origin_RL1,    "Original Image (RL=1)")
#show_img(0, img_origin_RL100,  "Original Image (RL=100)")
#show_img(0, img_noised_RL1,   "Noised Image (RL=1)")
#show_img(0, img_noised_RL10, "Noised Image (RL=10)")
#show_img(1, img_noised_RL20, "Noised Image (RL=20)")
# show_img(1, img_noised_RL30, "Noised Image (RL=30)")
# show_img(1, img_noised_RL40, "Noised Image (RL=40)")
# show_img(1, img_noised_RL50, "Noised Image (RL=50)")
# show_img(1, img_noised_RL60, "Noised Image (RL=60)")
# show_img(1, img_noised_RL70, "Noised Image (RL=70)")
# show_img(1, img_noised_RL80, "Noised Image (RL=80)")
# show_img(1, img_noised_RL90, "Noised Image (RL=90)")
# show_img(1, img_noised_RL100, "Noised Image (RL=100)")



# -------------------------------
# ----- Convert RGB to Gray -----
# -------------------------------
gray_img_origin_RL1   = cv2.cvtColor(img_origin_RL1,   cv2.COLOR_RGB2GRAY)
gray_img_origin_RL5   = cv2.cvtColor(img_origin_RL5,   cv2.COLOR_RGB2GRAY)
gray_img_origin_RL10  = cv2.cvtColor(img_origin_RL10,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL20  = cv2.cvtColor(img_origin_RL20,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL30  = cv2.cvtColor(img_origin_RL30,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL40  = cv2.cvtColor(img_origin_RL40,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL50  = cv2.cvtColor(img_origin_RL50,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL60  = cv2.cvtColor(img_origin_RL60,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL70  = cv2.cvtColor(img_origin_RL70,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL80  = cv2.cvtColor(img_origin_RL80,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL90  = cv2.cvtColor(img_origin_RL90,  cv2.COLOR_RGB2GRAY)
gray_img_origin_RL100 = cv2.cvtColor(img_origin_RL100, cv2.COLOR_RGB2GRAY)
gray_img_origin_RL150 = cv2.cvtColor(img_origin_RL150, cv2.COLOR_RGB2GRAY)
gray_img_origin_RL200 = cv2.cvtColor(img_origin_RL200, cv2.COLOR_RGB2GRAY)
gray_img_origin_RL300 = cv2.cvtColor(img_origin_RL300, cv2.COLOR_RGB2GRAY)
gray_img_origin_RL400 = cv2.cvtColor(img_origin_RL400, cv2.COLOR_RGB2GRAY)
gray_img_origin_RL500 = cv2.cvtColor(img_origin_RL500, cv2.COLOR_RGB2GRAY)

gray_img_noised_RL1   = cv2.cvtColor(img_noised_RL1,   cv2.COLOR_RGB2GRAY)
gray_img_noised_RL5   = cv2.cvtColor(img_noised_RL5,   cv2.COLOR_RGB2GRAY)
gray_img_noised_RL10  = cv2.cvtColor(img_noised_RL10,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL20  = cv2.cvtColor(img_noised_RL20,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL30  = cv2.cvtColor(img_noised_RL30,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL40  = cv2.cvtColor(img_noised_RL40,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL50  = cv2.cvtColor(img_noised_RL50,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL60  = cv2.cvtColor(img_noised_RL60,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL70  = cv2.cvtColor(img_noised_RL70,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL80  = cv2.cvtColor(img_noised_RL80,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL90  = cv2.cvtColor(img_noised_RL90,  cv2.COLOR_RGB2GRAY)
gray_img_noised_RL100 = cv2.cvtColor(img_noised_RL100, cv2.COLOR_RGB2GRAY)
gray_img_noised_RL150 = cv2.cvtColor(img_noised_RL150, cv2.COLOR_RGB2GRAY)
gray_img_noised_RL200 = cv2.cvtColor(img_noised_RL200, cv2.COLOR_RGB2GRAY)
gray_img_noised_RL300 = cv2.cvtColor(img_noised_RL300, cv2.COLOR_RGB2GRAY)
gray_img_noised_RL400 = cv2.cvtColor(img_noised_RL400, cv2.COLOR_RGB2GRAY)
gray_img_noised_RL500 = cv2.cvtColor(img_noised_RL500, cv2.COLOR_RGB2GRAY)
# print("gray_img_origin : ", gray_img_origin.shape)  
# print("gray_img_noised : ", gray_img_noised.shape)
# print("\n")



# -----------------------------------------------
# ----- Get statistical data of pixel value -----
# -----------------------------------------------
def get_data_of_pixel_value(_img, _img_name):
  print("===== Statistical Data of", _img_name, "(Gray) =====")
  print("Num of pixel values (== 255) :", np.sum(_img == 255))
  print("Num of pixel values (<= 1) :", np.sum(_img <= 1))
  print("Num of pixel values (== 0)   :", np.sum(_img == 0) )
  print("\nMax :", np.max(_img))
  print("Min :", np.min(_img))
  #print("\nAverage :", np.mean(_img))
  #print("Median  :", np.median(_img))
  print("\nAverage :", _img[_img != 0].mean())
  print("S.D.  :", _img[_img != 0].std())
  print("\n")
  
  return _img[_img != 0].mean()

# get_data_of_pixel_value(gray_img_origin)
# get_data_of_pixel_value(gray_img_origin_RL100)
# get_data_of_pixel_value(gray_img_noised)
# get_data_of_pixel_value(gray_img_noised_RL100)

# R_mean_RL1   = get_data_of_pixel_value(gray_img_origin_RL1,   "img_original_RL1")
# R_mean_RL5   = get_data_of_pixel_value(gray_img_origin_RL5,   "img_original_RL5")
# R_mean_RL10  = get_data_of_pixel_value(gray_img_origin_RL10,  "img_original_RL10")
# R_mean_RL20  = get_data_of_pixel_value(gray_img_origin_RL20,  "img_original_RL20")
# R_mean_RL30  = get_data_of_pixel_value(gray_img_origin_RL30,  "img_original_RL30")
# R_mean_RL40  = get_data_of_pixel_value(gray_img_origin_RL40,  "img_original_RL40")
# R_mean_RL50  = get_data_of_pixel_value(gray_img_origin_RL50,  "img_original_RL50")
# R_mean_RL60  = get_data_of_pixel_value(gray_img_origin_RL60,  "img_original_RL60")
# R_mean_RL70  = get_data_of_pixel_value(gray_img_origin_RL70,  "img_original_RL70")
# R_mean_RL80  = get_data_of_pixel_value(gray_img_origin_RL80,  "img_original_RL80")
# R_mean_RL90  = get_data_of_pixel_value(gray_img_origin_RL90,  "img_original_RL90")
# R_mean_RL100 = get_data_of_pixel_value(gray_img_origin_RL100, "img_original_RL100")
# R_mean_RL150 = get_data_of_pixel_value(gray_img_origin_RL150, "img_original_RL150")
# R_mean_RL200 = get_data_of_pixel_value(gray_img_origin_RL200, "img_original_RL200")
# R_mean_RL300 = get_data_of_pixel_value(gray_img_origin_RL300, "img_original_RL300")
# R_mean_RL400 = get_data_of_pixel_value(gray_img_origin_RL400, "img_original_RL400")
# R_mean_RL500 = get_data_of_pixel_value(gray_img_origin_RL500, "img_original_RL500")

R_mean_noised_RL1   = get_data_of_pixel_value(gray_img_noised_RL1,   "img_noised_RL1")
R_mean_noised_RL5   = get_data_of_pixel_value(gray_img_noised_RL5,   "img_noised_RL5")
R_mean_noised_RL10  = get_data_of_pixel_value(gray_img_noised_RL10,  "img_noised_RL10")
R_mean_noised_RL20  = get_data_of_pixel_value(gray_img_noised_RL20,  "img_noised_RL20")
R_mean_noised_RL30  = get_data_of_pixel_value(gray_img_noised_RL30,  "img_noised_RL30")
R_mean_noised_RL40  = get_data_of_pixel_value(gray_img_noised_RL40,  "img_noised_RL40")
R_mean_noised_RL50  = get_data_of_pixel_value(gray_img_noised_RL50,  "img_noised_RL50")
R_mean_noised_RL60  = get_data_of_pixel_value(gray_img_noised_RL60,  "img_noised_RL60")
R_mean_noised_RL70  = get_data_of_pixel_value(gray_img_noised_RL70,  "img_noised_RL70")
R_mean_noised_RL80  = get_data_of_pixel_value(gray_img_noised_RL80,  "img_noised_RL80")
R_mean_noised_RL90  = get_data_of_pixel_value(gray_img_noised_RL90,  "img_noised_RL90")
R_mean_noised_RL100 = get_data_of_pixel_value(gray_img_noised_RL100, "img_noised_RL100")
R_mean_noised_RL150 = get_data_of_pixel_value(gray_img_noised_RL150, "img_noised_RL150")
R_mean_noised_RL200 = get_data_of_pixel_value(gray_img_noised_RL200, "img_noised_RL200")
R_mean_noised_RL300 = get_data_of_pixel_value(gray_img_noised_RL300, "img_noised_RL300")
R_mean_noised_RL400 = get_data_of_pixel_value(gray_img_noised_RL400, "img_noised_RL400")
R_mean_noised_RL500 = get_data_of_pixel_value(gray_img_noised_RL500, "img_noised_RL500")



# ----------------------
# ----- Matplotlib -----
# ----------------------
# ----- originl image data -----
#ax[2].hist(gray_img_origin_RL1.ravel(), bins=50, color='red', alpha=0.5, label="Original Image (RL=1)")
#ax[2].hist(gray_img_origin_RL100.ravel(), bins=50, color='red', alpha=0.5, label="Original Image (RL=100)")

# ----- noised image data -----
ax[2].hist(img_noised_RL1[:, :, 0].ravel(), bins=50, color='red', alpha=0.5, label="Noised Image (RL=1)")
ax[2].hist(img_noised_RL100[:, :, 0].ravel(), bins=50, color='blue', alpha=0.5, label="Noised Image (RL=100)")
ax[2].axvline(R_mean_noised_RL1, color='red')
ax[2].axvline(R_mean_noised_RL100, color='blue')

ax[2].set_title("Histogram of Pixel Value of Gray Scale", fontsize=12)
ax[2].set_xlabel("Pixel value", fontsize=12)    # 画素値 
ax[2].set_ylabel("Number of pixels", fontsize=12) # 画素値の度数
ax[2].set_xlim([10, 266])
#ax[2].set_ylim([0, 250000])
#plt.grid()
ax[2].legend(fontsize=12)

# fig.show()
# plt.show()

