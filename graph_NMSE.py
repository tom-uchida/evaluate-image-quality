#
# NMSE
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



# -------------------------
# ----- Read csv file -----
# -------------------------
# original
LR_NMSE = pd.read_csv("NMSE.csv", header=None)

LR_NMSE = LR_NMSE.values

# 0, 10, 100, 200
#index = [0, 2, 11, 13]

LR   = LR_NMSE[:16,0]
NMSE = LR_NMSE[:16,1]
# std_original = original[:14,2]
# RL_original  = original[index,0]
# api_original = original[index,1]



# ----------------
# ----- Plot -----
# ----------------
# plt.plot(LR, NMSE, color='black')
c = plt.scatter(LR, NMSE, color='black')

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.xlabel('$L$', fontsize=14)
plt.ylabel('NMSE', fontsize=14) # Gray scale

# draw circle
# plt.scatter(RL_original[2], api_original[2], facecolor='none', s=200, edgecolor='black')
# plt.scatter(RL_noised[2], api_noised[2], facecolor='none', s=200, edgecolor='black')
# plt.scatter(RL_original[11], api_original[11], facecolor='none', s=200, edgecolor='black')
# plt.scatter(RL_noised[11], api_noised[11], facecolor='none', s=200, edgecolor='black')

# Draw text
# plt.text(5, 20.5, "20.5", fontsize=14, color='black')
# plt.text(10, 1.0, "0.336", fontsize=14, color='black')
# plt.text(100, 0.8, "0.0318", fontsize=14, color='black')

plt.xticks([1, 50, 100, 150], fontsize=14)
#plt.xticks([0, 10, 100])
plt.yticks([0, 5, 10, 15, 20], fontsize=14)
#plt.ylim([0, 275])

#plt.grid()
plt.legend(fontsize=12)

plt.show()