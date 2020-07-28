import numpy as np
import math
import cv2

# Check arguments
import sys
args = sys.argv
if len(args) != 3:
    print("\nUSAGE : $ python calc_NMSE.py [reference_image] [evaluation_image]")
    sys.exit()

def read_img(_img_name):
	# read input image
	img = cv2.imread(_img_name)

	# convert color (BGR → RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img

def PSNR(_img1_RGB, _img2_RGB):
    # Calc max pixel value
    MAX_PIXEL_VALUE = max(np.max(_img1_RGB), np.max(_img2_RGB))
    # print("\nMAX_1\n>", np.max(_img1_RGB))
    # print("\nMAX_2\n>", np.max(_img2_RGB))
    print("\nMAX_PIXEL_VALUE\n>", MAX_PIXEL_VALUE)

    # Convert RGB to Gray
    img1_gray = cv2.cvtColor(_img1_RGB, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(_img2_RGB, cv2.COLOR_RGB2GRAY)

    # Calc MSE
    # Grayscale ver.
    # MSE_gray = np.mean( (img1_gray - img2_gray) ** 2 ) # Gray
    # MSE_denominator = np.mean( (img1_gray) ** 2 )
    # MSE_normalized = MSE_gray / MSE_denominator
    # print("\nMSE_gray\n>", MSE_gray)
    # print("\nMSE_denominator\n>", MSE_denominator)
    # print("\nMSE_normalized\n>", MSE_normalized)
    # return 20 * math.log10(MAX_PIXEL_VALUE / math.sqrt(MSE_gray))

    # RGB color var.
    MN = 1000 ** 2
    MSE_rgb = 1/MN * np.sum( (_img1_RGB - _img2_RGB) ** 2 ) # RGB
    return 20*math.log10(MAX_PIXEL_VALUE) - 10*math.log10(MSE_rgb)

    # print("\nmse_R\n>", mse_R)
    # print("\nmse_G\n>", mse_G)
    # print("\nmse_B\n>", mse_B)
    # print("\nMSE_mean\n>", MSE_mean)
    # MSE = np.mean( (_img1_RGB - _img2_RGB) ** 2 )
    # print("\nMSE\n>", MSE)
    
    # return 20 * math.log10(MAX_PIXEL_VALUE / math.sqrt(MSE))

if __name__ == "__main__":
    # Read two images
    img_ref_I_RGB = read_img( args[1] )
    img_eva_K_RGB = read_img( args[2] )
    # print("\nPSNR =", PSNR(img_ref_I_RGB, img_eva_K_RGB), "\n")

    psnr = cv2.PSNR(img_ref_I_RGB, img_eva_K_RGB)
    print("\nPSNR =", psnr, "\n")