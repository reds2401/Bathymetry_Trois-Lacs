# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:06:30 2021

Bathymetry

Interpolation from bathymetry points for 2019 bathymetry using ordinary Kriging
and a pre-defined grid for projection.
An interpolation is made for each lake and then joined into a single dataset.

@author: reds2401
"""

# %% Libraries
import numpy as np
import pandas as pd
import pyproj as proj
from pykrige.rk import Krige

# %% Files

folder = './Bathy/Results_20211109(B2019)/'                # Folder with original bathymetry points
result_fol = './Bathy/Results_20211109(B2019)/'            # Folder to write results
p_grid = pd.read_csv('./Bathy/GPS_Sloped_WS_grid.csv')     # Projection grid
new_grid = p_grid[['id','X','Y','Lake']]
new_grid['Z_bed'] = np.nan
b_file = 'Bathy_2019_clean.csv'                     # File containing the bathymetry
file_df = pd.read_csv(folder + b_file)
bat_df = file_df[['ID','X','Y','Z_bed','Lake']]     # Columns to keep from file
bat_df.dropna(axis=0, inplace=True)

# %% Kriging calculations

for l in bat_df['Lake'].unique() :
    print('Kriging calculations for '+l)
    XY_train = np.array(bat_df[['X','Y']].loc[bat_df['Lake']==l])
    Z_train = np.array(bat_df['Z_bed'].loc[bat_df['Lake']==l])  # Elevation or depth for interpolation

    estimator = Krige(method='ordinary', variogram_model='spherical', nlags=100,    # Kriging parameters
                      n_closest_points=300, verbose=True,
                      coordinates_type='euclidean', pseudo_inv=True)
    estimator.fit(x=XY_train, y=Z_train)
    print(l+' Finished fit')

    # %% Kriging projection

    XY_pred = np.array(p_grid[['X', 'Y']].loc[p_grid['Lake']==l])   # X and Y grid
    Z_pred = estimator.predict(XY_pred)                             # Z or depth interpolated result column to add to the grid
    print('Correct prediction for '+l)

    # %% Writing new grid file
    new_grid['Z_bed'].loc[new_grid['Lake']==l] = Z_pred             # Assign bed elevation in new column of the new grid
filename = 'Result_' + b_file
new_grid.to_csv(result_fol + filename)                              # Write results file