import numpy as np
import math
import cv2

def read_img(_img_name):
	# read input image
	img = cv2.imread(_img_name)

	# convert color (BGR → RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img

# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL1.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR1.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL2.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR2.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL3.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR3.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL4.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR4.bmp")
img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL5.bmp")
img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR5.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL6.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR6.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL7.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR7.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL8.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR8.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL9.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR9.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL10.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR10.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL20.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR20.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL30.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR30.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL40.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR40.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL50.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR50.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL60.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR60.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL70.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR70.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL80.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR80.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL90.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR90.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL100.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR100.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL110.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR110.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL120.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR120.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL130.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR130.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL140.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR140.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL150.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR150.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL160.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR160.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL170.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR170.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL180.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR180.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL190.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR190.bmp")
# img_I_RGB = read_img("images/DATA/uniformly_new/1024/plane_10M_RL200.bmp")
# img_K_RGB = read_img("images/DATA/uniformly_new/1024_1e-05/gaussian_plane_LR200.bmp")

def NMSE(_img_I_RGB, _img_K_RGB):
    # Convert RGB to Gray
    img_I_gray = cv2.cvtColor(_img_I_RGB, cv2.COLOR_RGB2GRAY)
    img_K_gray = cv2.cvtColor(_img_K_RGB, cv2.COLOR_RGB2GRAY)

    # Calc NMSE
    MSE_numer = np.sum( (img_I_gray - img_K_gray) ** 2 )
    MSE_denom = np.sum( (img_I_gray) ** 2 )
    NMSE = MSE_numer / MSE_denom

    print("\nMSE_numer\n>", MSE_numer)
    print("\nMSE_denom\n>", MSE_denom)

    return NMSE

if __name__ == "__main__":
    print("NMSE =", NMSE(img_I_RGB, img_K_RGB))