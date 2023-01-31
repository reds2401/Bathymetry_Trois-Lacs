# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:06:30 2021

Bathymetry

Interpolation from bathymetry points using ordinary Kriging and a pre-defined
grid for projection. Nicolet river upstream of the lakes

@author: reds2401
"""

# %% Libraries
import numpy as np
import pandas as pd
import pyproj as proj
from pykrige.rk import Krige

import warnings
warnings.filterwarnings("ignore")

# %% Files

folder = './Bathy/Results_20220908(Nicolet_Upstream)/'                 # Folder with original bathymetry points
result_fol = './Bathy/Results_20220908(Nicolet_Upstream)/'             # Folder to write results
p_grid = pd.read_csv('./Bathy/Results_20220908(Nicolet_Upstream)/Grid_Nicolet_US.csv')   # Projection grid
new_grid = p_grid[['id','X','Y']]
new_grid['Depth'] = np.nan
b_file = 'Bathy_Nicolet_US.csv'             # File containing the bathymetry
file_df = pd.read_csv(folder + b_file)
bat_df = file_df[['Id','X','Y','Depth']]    # Columns to keep from file
bat_df.dropna(axis=0, inplace=True)

# %% Kriging calculations

XY_train = np.array(bat_df[['X','Y']])
Z_train = np.array(bat_df['Depth'])  # Elevation or depth for interpolation

estimator = Krige(method='ordinary', variogram_model='spherical', nlags=100,    # Kriging parameters
                  n_closest_points=100, verbose=True,
                  coordinates_type='euclidean', pseudo_inv=True)
estimator.fit(x=XY_train, y=Z_train)
print('Finished fit')

# %% Kriging projection

XY_pred = np.array(p_grid[['X', 'Y']].astype(float))        # X and Y grid
D_pred = estimator.predict(XY_pred)                         # Z or depth interpolated result column to add to the grid
print('Correct prediction found')
D_pred[D_pred < 0] = 0                                      # Forcing all negative values to zero


# %% Writing new grid file
new_grid['Depth'] = D_pred
new_grid['Z_bed'] = p_grid['Z'] - D_pred                    # Calcul bed elevation in new column of the new grid
filename = 'Result_' + b_file
new_grid.to_csv(result_fol + filename)                      # Write results file