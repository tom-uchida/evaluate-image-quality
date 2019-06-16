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
    # Read two images
    img_ref_I_RGB = read_img( args[1] )
    img_eva_K_RGB = read_img( args[2] )

    # Calc NMSE
    print("\nNMSE =", NMSE(img_ref_I_RGB, img_eva_K_RGB), "\n")