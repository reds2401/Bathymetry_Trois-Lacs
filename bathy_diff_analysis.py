# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:48:29 2022

Analysis of bathymetry differences (obtained with QGIS) against base bathymetry of 2019

@author: reds2401
"""

# %% Libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# %% Files

folder = 'C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Maps/QGIS_Model/Bathymetry/'
folder_list = os.listdir(folder)
b_files = [b for b in folder_list if ('BathyDiff' in b in b)]   # Only read the bathymetry differences files

# %% Build dataframe

bat_dif_df = pd.read_csv(folder + b_files[0])[['X','Y','Lake']]         # Initialize a dataframe for all Z differences

for file in b_files:
    col_name = file[15:-4]                                              # Name of each bathymetry
    b_file = pd.read_csv(folder + file)                                 # Read Bathymetry file
    bat_dif_df[col_name] = b_file['Zb_diff']

# %% Plot Histogram

b_hist_df = bat_dif_df[['1974','2004','2020.11','2021.05','2021.8.5','2021.8.6']]
b_hist_w = 100*np.ones(b_hist_df.shape) / b_hist_df.shape[0]                                # Histogram weights
plt.figure(dpi = 150,figsize=(7,7))                                                         # Initialize figure
b_n, b_bins, b_fills = plt.hist(b_hist_df, bins = np.arange(-2,5,0.5), weights = b_hist_w,
                                  rwidth=0.9, fill=True, linewidth=0.5)                     # Calcul figure parameters

# Format figure
fig_colors = ['turquoise', 'aquamarine', 'deepskyblue', 'royalblue', 'mediumblue',  'navy']
for b_fill_set, color in zip(b_fills, fig_colors):
    for b_fill in b_fill_set.patches:
        b_fill.set_color(color)
plt.xlabel('Elevation difference (m)')
plt.ylabel('Percentage of bathymetry points')
plt.title('Bathymetry comparison against 2019 data')
plt.grid(axis='y', alpha=0.75)

plt.legend(b_hist_df.columns, fontsize = 'small')                                           # Print figure



# %% In case of using hatches instead of colors
#hatches = ['xx', '----', '', '....', '******',  '////']
#for patch_set, hatch in zip(b_patches, hatches):
#    for patch in patch_set.patches:
#        patch.set_hatch(hatch)
# bat_df = bat_df.dropna(axis = 0)