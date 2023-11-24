# coding=utf-8

#******************************************************************************
#
# Data
# 
# version: 1.0
# Copyright: (c) October 2023
# Authors: Ernesto Giron (e.giron.e@gmail.com)
#
#
# This source is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# A copy of the GNU General Public License is available on the World Wide Web
# at <http://www.gnu.org/copyleft/gpl.html>. You can also obtain it by writing
# to the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
# MA 02111-1307, USA.
#
#******************************************************************************

from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


import os, sys
import pandas as pd
import datetime as dt
from sklearn.neighbors import NearestNeighbors

def load_dataset(name='kansas'):
    '''
        Load example datasets

        Return:
            A data dictionary with all raw data and information needed to carry out the demo.


        Examples:
            ``` raw
                >>> from pywheat.data import load_dataset
                >>> # Load example dataset (Kansas State Univ. example data)
                >>> data = load_dataset('kansas')
                >>> print(data.keys()) # dict_keys(['Weather'])
                >>> # Display Weather data
                >>> data['Weather']
            ```        
    '''
    file_path = os.path.realpath(__file__)
    example_data_path = os.path.join(file_path.replace('__init__.py',''), 'example')
    
    if (name=='kansas'):
        print("Loading example weather dataset \nfrom Kansas State University (Wagger,M.G. 1983) stored in DSSAT v4.8.")
        weather = pd.read_parquet(os.path.join(example_data_path, "weather_Kansas_State_Univ_WaggerMG_1983.parquet"))
    
    data = {
        "Weather":weather
    }
    return data

# ---------------------------
# Load configuration files
# ---------------------------
def load_configfiles():
    file_path = os.path.realpath(__file__)
    config_data_path = os.path.join(file_path.replace('__init__.py',''), 'config')
    
    configbycoords = None
    configbygenotype = None
    nn = None
    configbycoords_path = os.path.join(config_data_path, 'configbycoords.cfg')
    configbygenotype_path = os.path.join(config_data_path, 'configbygenotype.cfg')
    if os.path.exists(config_data_path):
        if os.path.isfile(configbycoords_path):
            configbycoords = pd.read_parquet(configbycoords_path)
            # Setup a model
            nn = NearestNeighbors()#metric="haversine"
            nn.fit(configbycoords[["lat", "lon", 'sowing_date']])
        if os.path.isfile(configbygenotype_path):
            configbygenotype = pd.read_parquet(configbygenotype_path)
    
    config = {
        "nn": nn,
        "configbycoords":configbycoords,
        "configbygenotype": configbygenotype
    }
    return config