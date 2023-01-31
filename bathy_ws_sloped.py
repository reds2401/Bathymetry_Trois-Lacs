# -*- coding: utf-8 -*-
"""
Created on Fri Oct 1 2021

Bathymetry

Building a grid of water elevation for Trois-Lacs with a constant slope along
the axe X.

@author: reds2401
"""

# %% Libraries
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

# %% Files

folder = 'C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Code/Bathy/Data_20211020/'  # Folder with ADCP and GPS result datasets
p_grid = pd.read_csv('C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Code/Bathy/Lake_grid.csv')  # Projection grid
file_list = os.listdir(folder)  # File list
bathy_dates = {'2020_11': [0.0000287, 158.009],'2021_05': [0.0000287, 158.009],
               '2021_08_05': [0.00003414, 156.683],'2021_08_06': [0.00003122, 157.259]}  # Dates of the different bathymetries

# %% Sloped grid

for b in bathy_dates.keys() :
    p_grid['Z_'+b] = p_grid['X']*bathy_dates[b][0]+bathy_dates[b][1]

p_grid.to_csv('GPS_Sloped_WS_grid.csv')