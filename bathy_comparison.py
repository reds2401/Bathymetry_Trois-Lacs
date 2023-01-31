# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:57:13 2021

Comparison of bathymetry results:
    Calculation of errors
    Calculation of volumes and average depth
@author: reds2401
"""

# %% Libraries
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2

# %% Files

folder = 'C:/Users/reds2401/OneDrive - USherbrooke/Research_project/Code/Bathy/Results_complete/'
folder_list = os.listdir(folder)
b_files = [b for b in folder_list if ('Res' in b in b)]     # List result bathymetry files

bat_dep_df = pd.read_csv(folder + b_files[0])[['X','Y']]    # Get coordinates from first bathy file

for file in b_files:                                        # Build a DF with all Z elevations of the bathymetries
    b_name = file[13:-4]
    b_file = pd.read_csv(folder + file)
    bat_dep_df[b_name] = b_file['Z_bed']                    # Add bathymetry's Z elevation column to the Bathy DataFrame

bat_dep_df = bat_dep_df.dropna(axis = 0)                    # Delete rows without NA values
bat_df = bat_dep_df.copy()                                  # Duplicate DF (for modification)
bat_df.iloc[:,2::] = 164 - bat_dep_df.iloc[:,2::]           # Calcul Depth values for all bathymetries
# %% Calculation of errors

# Creating dataframes for the different errors
mbe_df = pd.DataFrame(index = bat_df.columns[2::], columns = bat_df.columns[2::])       # Mean Biased Error
mae_df = pd.DataFrame(index = bat_df.columns[2::], columns = bat_df.columns[2::])       # Mean Absolute Error
mse_df = pd.DataFrame(index = bat_df.columns[2::], columns = bat_df.columns[2::])       # Mean Square Error
rmse_df = pd.DataFrame(index = bat_df.columns[2::], columns = bat_df.columns[2::])      # Root-Mean Square Deviation
mape_df = pd.DataFrame(index = bat_df.columns[2::], columns = bat_df.columns[2::])      # Mean Absolute Percentage Error
r2_df = pd.DataFrame(index = bat_df.columns[2::], columns = bat_df.columns[2::])        # Coefficient of determination

# Calcul of errors for all bathymetries
for lin in bat_df.columns[2::]:
    for col in bat_df.columns[2::]:
        mbe_df.loc[lin, col] = (bat_df[lin]-bat_df[col]).mean()
        mae_df.loc[lin, col] = mae(bat_df[lin], bat_df[col])
        mse_df.loc[lin, col] = mse(bat_df[lin], bat_df[col], squared=False)
        rmse_df.loc[lin, col] = mse(bat_df[lin], bat_df[col], squared=True)
        mape_df.loc[lin, col] = mape(bat_df[lin], bat_df[col])
        r2_df.loc[lin, col] = r2(bat_df[lin], bat_df[col])

# %% General metrics calculation
idx = ['Water_volume', 'Sediment_volume','Average_elevation']                               # Metrics to calcul
gm_df = pd.DataFrame(index = idx, columns=bat_df.columns[2::])                          # Create DF to contain metrics

# Calcul bathymetry metrics
for b in bat_df.columns[2::]:
    gm_df.loc[idx[0]][b] = round(bat_df[b].sum()*100/1000000,3)                         # Total water volume
    gm_df.loc[idx[1]][b] = round((bat_df[b]-bat_df['2019']).sum()*100/1000000,3)        # Total sediment volume
    gm_df.loc[idx[2]][b] = round(bat_dep_df[b].mean(),2)                                # Average elevation

