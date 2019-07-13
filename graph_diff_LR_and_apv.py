#
# リピートレベルと平均輝度値の差のを表すグラフを作成するプログラム
#

import numpy as np
import pandas as pd
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
    print("\nUSAGE : $ python graph_LR_and_apv.py [original_avg_pixel_value.cav] [noise_avg_pixel_value.csv]")
    sys.exit()

# Read csv file
df_original = pd.read_csv( args[1], header=None )
df_noise    = pd.read_csv(args[2] , header=None )

original    = df_original.values
noise       = df_noise.values

# 0, 10, 100, 200
index = [0, 2, 11, 13]

LR_original  = original[:16,0]
apv_original = original[:16,1]
# LR_original  = original[:21,0]
# apv_original = original[:21,1]
# LR_original  = original[:11,0]
# apv_original = original[:11,1]

LR_noise    = noise[:16,0]
apv_noise   = noise[:16,1]
# LR_noise    = noise[:21,0]
# apv_noise   = noise[:21,1]
# LR_noise    = noise[:11,0]
# apv_noise   = noise[:11,1]

# Calc diff b/w Original Image and Noise Image
diff = apv_original - apv_noise
diff_apv = np.abs(diff)
print(diff_apv)

# Figure
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# diff
plt.scatter(LR_original, diff_apv, color='black')
# plt.ylim([-5, 100])

plt.xlabel('$L$', fontsize=14)
plt.ylabel('Difference of average pixel value', fontsize=14) # Diff

plt.xticks([1, 10, 100, 150])
# plt.xticks([1, 50, 100, 150, 200])
# plt.xticks([1, 50, 100])
plt.yticks([0, 50, 100, 150, 200, 250])

plt.grid()

plt.show()