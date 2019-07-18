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
if len(args) != 2:
    print("\nUSAGE   : $ python graph_MSE.py [csv_file_path]")
    sys.exit()


# Read csv file
csv = pd.read_csv(args[1], header=None)

# Convert to numpy
LR_MSE = csv.values
# LR  = LR_MSE[0:11,0] # L=1-100
# MSE = LR_MSE[0:11,1] 
LR  = LR_MSE[0:16,0] # L=1-150
MSE = LR_MSE[0:16,1]

# Create figure
c = plt.scatter(LR, MSE, color='black')

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.xlabel('$L$', fontsize=14)
plt.ylabel('MSE', fontsize=14) # Gray scale

# plt.xticks([1, 50, 100, 150], fontsize=14)
plt.xticks([1, 10, 100, 150], fontsize=14)
# plt.xticks([1, 50, 100], fontsize=14)
# plt.yticks([0, 5000, 10000, 15000, 20000], fontsize=12)
plt.yticks([0, 5000, 10000, 15000, 20000], fontsize=12)

plt.grid()
plt.legend(fontsize=14)

plt.show()