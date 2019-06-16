import numpy as np
import math
import cv2

def read_img(_img_name):
	# read input image
	img = cv2.imread(_img_name)

	# convert color (BGR → RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img

# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL1.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR1.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL5.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR5.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL10.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR10.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL20.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR20.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL30.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR30.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL40.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR40.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL50.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR50.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL60.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR60.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL70.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR70.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL80.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR80.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL90.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR90.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL100.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR100.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL110.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR110.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL120.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR120.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL130.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR130.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL140.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR140.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL150.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR150.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL160.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR160.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL170.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR170.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL180.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR180.bmp")
img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL190.bmp")
img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR190.bmp")
# img1_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL200.bmp")
# img2_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR200.bmp")

def psnr(_img1_RGB, _img2_RGB):
    # Calc max pixel value
    MAX_PIXEL_VALUE = max(np.max(_img1_RGB), np.max(_img2_RGB))
    # print("\nMAX_1\n>", np.max(_img1_RGB))
    # print("\nMAX_2\n>", np.max(_img2_RGB))
    print("\nMAX_PIXEL_VALUE\n>", MAX_PIXEL_VALUE)

    # Convert RGB to Gray
    img1_gray = cv2.cvtColor(_img1_RGB, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(_img2_RGB, cv2.COLOR_RGB2GRAY)

    # Calc MSE
    MSE_gray = np.mean( (img1_gray - img2_gray) ** 2 ) # Gray
    MSE_denominator = np.mean( (img1_gray) ** 2 )
    MSE_normalized = MSE_gray / MSE_denominator
    print("\nMSE_gray\n>", MSE_gray)
    print("\nMSE_denominator\n>", MSE_denominator)
    print("\nMSE_normalized\n>", MSE_normalized)

    return 20 * math.log10(MAX_PIXEL_VALUE / math.sqrt(MSE_gray))

    # print("\nmse_R\n>", mse_R)
    # print("\nmse_G\n>", mse_G)
    # print("\nmse_B\n>", mse_B)
    # print("\nMSE_mean\n>", MSE_mean)
    # MSE = np.mean( (_img1_RGB - _img2_RGB) ** 2 )
    # print("\nMSE\n>", MSE)
    
    # return 20 * math.log10(MAX_PIXEL_VALUE / math.sqrt(MSE))

db = psnr(img1_RGB, img2_RGB)

# print("\nPSNR\n>", db)
print("\n")