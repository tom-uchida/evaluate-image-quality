# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.style.use('seaborn-whitegrid')

from matplotlib import cycler
# colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
# plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=False, prop_cycle=colors)
# plt.rc('grid', color='w', linestyle='solid')
#plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=1)

plt.figure(figsize=(8,6))

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
df_noised = pd.read_csv("LR_mpv_gaussian_1e-05_1024.csv", header=None)

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
print(diff_api)



# ----------------
# ----- Plot -----
# ----------------
# original
plt.plot(RL_original, api_original, color='blue', label="Ground truth")
plt.scatter(RL_original, api_original, color='blue', cmap='gray', marker="o")

# # gaussian
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["font.size"] = 12
# plt.plot(RL_noised, api_noised, color='red', label="Gaussian noise, $σ^2=0.01$")
plt.plot(RL_noised, api_noised, color='red', label=r"Gaussian noise, $σ^2_{\rm init}=1.0 \times 10^{-5}$")
plt.scatter(RL_noised, api_noised, color='red', cmap='gray', marker=",")

# error bar
# plt.errorbar(RL_original, api_original, yerr=std_original, fmt='ro', ecolor='blue')
# plt.errorbar(RL_noised, api_noised, yerr=std_noised, fmt='ro', ecolor='red')

# diff
# plt.plot(RL_original, diff_api, color='black')
# plt.scatter(RL_original, diff_api, color='black')
# plt.text(5, 89, "90", fontsize=12, color='black')
# plt.text(13, 15, "15", fontsize=12, color='black')
# plt.text(103, 3, "1", fontsize=12, color='black')
# plt.ylim([-5, 100])


plt.xlabel('Repeat level : $L_{\mathrm{R}}$', fontsize=12)
plt.ylabel('Mean pixel value', fontsize=12) # Gray scale
#plt.ylabel('Difference of mean pixel value', fontsize=12) # Diff

# Text
plt.text(-10, 240, "255", fontsize=12, color='b')
plt.text(-10, 150, "165", fontsize=12, color='r')
plt.text(15, 192-1, "192", fontsize=12, color='b')
plt.text(15, 174-1, "177", fontsize=12, color='r')
plt.text(104, 45, "41", fontsize=12, color='b')
plt.text(104, 26, "40", fontsize=12, color='r')

plt.xticks([1, 10, 100, 200])
plt.yticks([50, 100, 150, 200, 255])
plt.xlim([-10, 110])

#plt.grid()
plt.legend(fontsize=12)

plt.show()