#
# リピートレベルと平均輝度値の関係を表すグラフを作成するプログラム
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
df_original = pd.read_csv("RL_and_avg_pixel_value_original_1024.csv", header=None)

original = df_original.values
print(original.shape)

# 0, 10, 100, 200
index = [0, 2, 11, 13]

RL_original  = original[:14,0]
api_original = original[:14,1]
# std_original = original[:14,2]
# RL_original  = original[index,0]
# api_original = original[index,1]

# gaussian
df_noised = pd.read_csv("RL_and_avg_pixel_value_gaussian_1024.csv", header=None)

noised = df_noised.values
print(noised.shape)

RL_noised    = noised[:14,0]
api_noised   = noised[:14,1]
# std_noised   = noised[:14,2]
# RL_noised    = noised[index,0]
# api_noised   = noised[index,1]



# -------------------------------------------------------
# ----- Calc diff b/w Ground Truth and Gaussian 10% -----
# -------------------------------------------------------
diff = api_original - api_noised
diff_api = np.abs(diff);
# print(diff_api)



# ----------------
# ----- Plot -----
# ----------------
# original
plt.plot(RL_original, api_original, color='blue', label="Ground truth")
c = plt.scatter(RL_original, api_original, color='blue', cmap='gray')

# # gaussian
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.plot(RL_noised, api_noised, color='red', label=r"Gaussian noise, $σ^2_{\rm init}=0.01$")
plt.scatter(RL_noised, api_noised, color='red', cmap='gray')

# error bar
# plt.errorbar(RL_original, api_original, yerr=std_original, fmt='ro', ecolor='blue')
# plt.errorbar(RL_noised, api_noised, yerr=std_noised, fmt='ro', ecolor='red')

# diff
# plt.plot(RL_original, diff_api, color='black')
# plt.scatter(RL_original, diff_api, color='black')
# plt.text(5, 90, "91", fontsize=12, color='black')
# plt.text(14, 20, "18", fontsize=12, color='black')
# plt.text(104, 4, "2", fontsize=12, color='black')
# plt.ylim([-5, 100])


plt.xlabel('Repeat level:$L_{\mathrm{R}}$', fontsize=12)
plt.ylabel('Mean pixel value', fontsize=12) # Gray scale
# plt.ylabel('Difference of mean pixel value', fontsize=12) # Diff

# draw circle
# plt.scatter(RL_original[2], api_original[2], facecolor='none', s=200, edgecolor='black')
# plt.scatter(RL_noised[2], api_noised[2], facecolor='none', s=200, edgecolor='black')
# plt.scatter(RL_original[11], api_original[11], facecolor='none', s=200, edgecolor='black')
# plt.scatter(RL_noised[11], api_noised[11], facecolor='none', s=200, edgecolor='black')

# Text
plt.text(-10, 240, "255", fontsize=12, color='b')
plt.text(-10, 150, "164", fontsize=12, color='r')
plt.text(15, 192-1, "192", fontsize=12, color='b')
plt.text(15, 174-1, "174", fontsize=12, color='r')
plt.text(104, 45, "41", fontsize=12, color='b')
plt.text(104, 25, "39", fontsize=12, color='r')

plt.xticks([1, 10, 100, 200])
#plt.xticks([0, 10, 100])
plt.yticks([50, 100, 150, 200, 255])

plt.grid()
plt.legend(fontsize=12)

plt.show()