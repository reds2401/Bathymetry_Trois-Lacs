# -*- coding: utf-8 -*-
"""
Created on Fri Oct 1 2021

Bathymetry

Interpolation from bathymetry points for multiple files using ordinary Kriging
and a pre-defined grid for projection.
An interpolation is made for each lake and then joined into a single dataset.
Results are produced for both types of beam measurements of the ADCP.

@author: reds2401
"""

# %% Libraries
import numpy as np
import pandas as pd
import os
import pyproj as proj
from pykrige.rk import Krige

import warnings
warnings.filterwarnings("ignore")

# %% Files

folder = './Bathy/Data_20211022/'  # Folder with ADCP result datasets
res_fol = './Bathy/Results_20211109/' # Folder to write results
p_grid = pd.read_csv('./Bathy/GPS_Sloped_WS_grid.csv')  # Projection grid
l_limit = pd.read_csv('./Bathy/Lake_shore_points.csv')[['X','Y','Depth', 'Lake']]
l_limit.columns = ['X','Y','Z','L']
file_list = os.listdir(folder)  # File list
bathy_dates = ['2020_11','2021_05','2021_08_05','2021_08_06']  # Dates of the different bathymetries
bf_dict = {}  # Initializating dictionary containing the respective filenames for the specific dates
for b_date in bathy_dates :
    bf_dict[b_date] = [f for f in file_list if b_date in f]  # Building the dictionary

# %% ADCP File: Bathy dates

for b in bathy_dates :
    adcp_file = pd.read_csv(folder + bf_dict[b][0])  # Read ADCP file
    adcp_file['Depth_SBA'] = adcp_file[['Depth_SB1','Depth_SB2','Depth_SB3','Depth_SB4']].mean(axis=1)  # Calcul mean SBA depth
    depth_types = ['Depth_VB','Depth_SBA']                                                              # Beam depth types

# %% Removing extreme differences between beams
    adcp_file['Beam_diff'] = abs(adcp_file['Depth_SBA'] - adcp_file['Depth_VB'])
    adcp_file['Perc_diff'] = adcp_file['Beam_diff']/(adcp_file[['Depth_VB', 'Depth_SBA']]).max(axis=1)
    adcp_file['Delete_Condition'] = (adcp_file.Beam_diff > 1) | (adcp_file.Perc_diff > 0.7)
    adcp_df_clean = adcp_file.drop(adcp_file[adcp_file.Delete_Condition == True].index, inplace=False) # ADCP DataFrame for points without big differences

# %% Coordinate transformation
    transformer = proj.Transformer.from_crs(4326, 32187)  # Object to transform the coordinates
    adcp_xtr, adcp_ytr = transformer.transform(np.array(adcp_df_clean['Latitude']), np.array(adcp_df_clean['Longitude']))  # Coordinates transform to euclidean

# %% Cycle thru Beam Depth types
    for d in depth_types :
        adcp_df = pd.DataFrame(np.array([adcp_xtr, adcp_ytr, adcp_df_clean[d], adcp_df_clean['Lake']]).T, columns = ['X','Y','Z','L'])  # ADCP Dataframe of each bathymetry and beam type
        adcp_df.dropna(inplace=True)                                             # Removing NaN fields
        adcp_df = pd.concat([adcp_df, l_limit])                                  # Adding the edge points where Depth = 0

        new_grid = p_grid[['id','X','Y','Lake','Z_'+b]]                          # New grid from projection of the bathymetries
        new_grid['Depth'] = np.nan

# %% Kriging interpolation for each Lake
        for l in adcp_df['L'].unique() :

        # %% ADCP Kriging
            int_id = 'Bathymetry_'+b+'_BeamType_'+d+'_'+l                              # ID of the bathymetry and type of beam being calculated
            print('Working on ' + int_id)

            estimator = Krige(method='ordinary', variogram_model='spherical', nlags=100,    # Kriging parameters
                              n_closest_points=300, coordinates_type='euclidean',
                              exact_values=True)

            XY_train = np.array(adcp_df[['X','Y']].loc[adcp_df['L']==l])        # Training coordinate data
            Z_train = np.array(adcp_df['Z'].loc[adcp_df['L']==l])               # Training elevation data
            estimator.fit(x=XY_train, y=Z_train)                                # Perform fit of the training dataset
            print('Finished fit')

            # %% ADCP Kriging projection
            XY_pred = np.array(new_grid[['X', 'Y']].loc[new_grid['Lake']==l])   # Prediction coordinate data
            Z_pred = estimator.predict(XY_pred)                                 # Perform prediction on the New Grid
            Z_pred = np.where(Z_pred<0, 0, Z_pred)                              # Remove negative values with Zeros
            print('Correct prediction for '+int_id)

            new_grid['Depth'].loc[new_grid['Lake']==l] = Z_pred                 # Insert depth prediction column to New Grid for each lake

        # %% Writing new grid file
        new_grid['Z_bed'] = new_grid['Z_'+b] - new_grid['Depth']                # Calcul Bed elevation as a new column
        new_grid.to_csv(res_fol + 'Bathy_Result-' + b+'_'+d + '.csv')           # Write results file