# coding=utf-8
#******************************************************************************
#
# Python scripts to help PyWheat processing Sentinel-2, MODIS, LandSat and ERA5 datasets.
#
# Copyright (C) 2023 Ernesto Giron (e.giron.e@gmail.com)
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

import os, gc, time, glob
import pathlib
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec
#from matplotlib.collections import PathCollection
#import matplotlib.dates as mdates
import seaborn as sns

# To use with GEE functions
#import ee
import xarray
import concurrent
import io
import json
import multiprocessing
import requests
#import tensorflow as tf
from google.api_core import retry
#from google.colab import auth
from google.protobuf import json_format
#from matplotlib import rc
from tqdm.notebook import tqdm
from IPython.display import Image
from numpy.lib import recfunctions as rfn


# TODOs: 
# - Create function to validate GEE auth 

def getERA5(init_dates=None, coords=None, agg='DAILY_AGGR', buffer=1, dispFig=False):
    ''' Extract weather values from ERA5 dataset
    
        Parameters:
            init_dates (array): Array of arrays of initial dates. Values in string format 'YYYY-MM-DD'. eg. ['1972-03-13']
            coords (array): Array of arrays of geographic coordinates in longitude, latitude. ie. [[83.45,27.5]]
            agg (str): Name of the collection to use in Google Earth Engine (GEE). Default value is 'DAILY_AGGR'
            buffer (int): Buffer in meters to extract pixels values around the ROI
            dispFig (bool): Display a figure with the results. 
            
        Usage:
            ``` python
                sowing_date = '2019-01-28'
                lng, lat = 83.45, 27.5
                final_arrays = getERA5(init_dates=[sowing_date], coords=[[lng,lat]],
                                       agg='DAILY_AGGR', buffer=1, dispFig=True)
            ```
        
        Returns:
            final_era5 (array): A table or pandas dataframe with all features extracted from ERA5 
            
    '''
    if (init_dates is None):
        print("Initial dates are not valid")
        return
    if (coords is None):        
        print("Coordinates not valid")
        return
    
    # Init variables
    assert len(coords)==len(init_dates)
    FEATURES = [] #total_precipitation_hourly
    BUFFER = buffer # meter
    SCALE = 11132 # Output resolution in meters.
    # Pre-compute a geographic coordinate system.
    proj = ee.Projection('EPSG:4326').atScale(SCALE).getInfo()
    # Get scales in degrees out of the transform.
    SCALE_X = proj['transform'][0]
    SCALE_Y = -proj['transform'][4]
    
    if (agg=='DAILY'):
        SCALE = 27830
    elif (agg=='DAILY_AGGR'):
        SCALE = 11132
    elif (agg=='MONTHLY'):
        SCALE = 11132
    elif (agg=='MONTHLY_AGGR'):
        SCALE = 11132
    elif (agg=='HOURLY'):
        SCALE = 11132
    
    if (agg=='DAILY'):
        # Temp in K and prec in meters
        FEATURES = ['mean_2m_air_temperature', 'minimum_2m_air_temperature', 'maximum_2m_air_temperature',
                   'total_precipitation', 'u_component_of_wind_10m']
    elif (agg=='DAILY_AGGR'):
        # Temp in K and prec in meters
        FEATURES = ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'total_precipitation_sum'] 
        #skin_temperature, soil_temperature_level_1 - 4, snow_depth, volumetric_soil_water_layer_1 - 4, 
        # surface_net_solar_radiation_sum, leaf_area_index_high_vegetation, leaf_area_index_low_vegetation
    elif (agg=='MONTHLY'):
        # ERA5 Monthly Aggregates - Latest Climate Reanalysis Produced by ECMWF / Copernicus Climate Change Service
        FEATURES = ['mean_2m_air_temperature', 'minimum_2m_air_temperature', 
                    'maximum_2m_air_temperature','total_precipitation' ]
    elif (agg=='MONTHLY_AGGR'):
        # ERA5-Land Monthly Aggregated - ECMWF Climate Reanalysis
        FEATURES = ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'total_precipitation_sum'] 
    elif (agg=='HOURLY'):
        # ERA5-Land Hourly - ECMWF Climate Reanalysis
        FEATURES = ['temperature_2m', 'total_precipitation', 'total_precipitation_hourly']
    else:
        FEATURES = []
        print("None of the bands were selected")
        return
    # Overwrite features
    #if (var is not None):
    #    FEATURES = var
    
    @retry.Retry()
    def get_patch(coords, asset_id, bands, buf=1):
        """Get a patch of pixels from an asset, centered on the coords."""
        point = ee.Geometry.Point(coords)
        request = {
            'fileFormat': 'NPY',
            'bandIds': bands,
            'region': point.buffer(buf).bounds().getInfo(), # Get one meter around by default
            'assetId': asset_id
        }
        return np.load(io.BytesIO(ee.data.getPixels(request))) #[band] return all bands
    
    # Creamos funciones para procesar muchos puntos en paralelo 
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=200)
    
    def setup_collection(sowing_date, region, agg='DAILY', bands=[]):
        sowing = ee.Date(sowing_date)
        start = sowing.advance(-1, 'month')
        end = sowing.advance(1, 'year')
        if (agg=='DAILY'):
            col = 'ECMWF/ERA5/DAILY'
        elif (agg=='DAILY_AGGR'):
            col = 'ECMWF/ERA5_LAND/DAILY_AGGR'
        elif (agg=='MONTHLY'):
            col = 'ECMWF/ERA5/MONTHLY'
        elif (agg=='MONTHLY_AGGR'):
            col = 'ECMWF/ERA5/MONTHLY_AGGR'
        elif (agg=='HOURLY'):
            col = 'ECMWF/ERA5_LAND/HOURLY'
        else:
            print("Collection not found")
            return None
        
        imgCol = ee.ImageCollection(col).select(bands)
        filter_imgCol = imgCol.filterBounds(region).filterDate(start, end)
        return filter_imgCol.getInfo()['features']
    
    # Process sites by sowing date and coordinates
    final_arrays = []
    for i, (point, sowing_date) in tqdm(enumerate(zip(coords, init_dates)), total=len(coords)):
        #print(i, sd, point)
        #sowing_date = init_dates[i], point = coords[i]
        region = ee.Geometry.Point(point) # Region of interest.
        fts = setup_collection(sowing_date, region, agg, FEATURES)
        future_to_image = {
            executor.submit(get_patch, point, image['id'], FEATURES, BUFFER):
                image['id'] for image in fts
        }
        
        arrays = ()
        types = []
        for future in concurrent.futures.as_completed(future_to_image):
            image_id = future_to_image[future]
            image_name = image_id.split('/')[-1]
            try:
                np_array = future.result()
                # Convert features in different units
                arrays += (np_array,)
                types.append((image_name, np.object_, np_array.shape))
            except Exception as e:
                print(e)
                pass
        final_array = np.array([arrays], types) # arrays with size of the features
        final_arrays.append({"id":i+1, "values":final_array})
    #
    # Extract ERA5 values by date
    era5_values = {}
    for fa in final_arrays:
        _id = fa['id']
        final_values = fa['values']
        for f in final_values.dtype.names:
            d = dt.datetime.strptime(f, "%Y%m%d")
            try:
                l = len(final_values[f].flatten())
                for v in final_values[f].flatten():
                    #'temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'total_precipitation_sum'
                    tavg, tmin, tmax, prec = v
                    tavg += tavg # daily mean temperature_2m. Units in K
                    tmin += tmin # daily minimum temperature_2m
                    tmax += tmax # daily maximum temperature_2m
                    prec += prec # The units of precipitation are depth in meters.
                # Average all observations or pixels and convert to required units
                tavg = round((tavg/l) - 273.15, 2) # Convert Kelvin to degree Celsius
                tmin = round((tmin/l) - 273.15, 2)
                tmax = round((tmax/l) - 273.15, 2)
                prec = round((prec/l) * 1000, 1)  # Convert meters to milimeters
                #print(tavg, tmin, tmax, prec )
            except Exception as err:
                print("Ploblem processing weather data", err)

            img_d = d.strftime('%Y-%m-%d')
            year = int(d.strftime('%Y'))
            month = str(d.strftime('%b'))
            doy = int(d.strftime('%j'))
            era5_values[f"{_id}_{f}"] = {"id":_id,"image_id":f,"date":img_d, "year":year, "month":month, "doy":doy, 
                                         "Tmin":tmin, "Tavg": tavg, "Tmax":tmax, "Prec":prec }

    #ndvi_values
    final_era5 = pd.DataFrame(era5_values).T.reset_index(drop=True)
    final_era5[["Tavg", "Tmin", "Tmax", "Prec"]] = final_era5[["Tavg", "Tmin", "Tmax", "Prec"]].astype(float)
    final_era5["year"] = final_era5["year"].astype(int)
    final_era5["doy"] = final_era5["doy"].astype(int)
    final_era5["date"] = pd.to_datetime(final_era5["date"].astype(str), format='%Y-%m-%d')
    final_era5 = final_era5.sort_values(["id","date"])
    
    if (dispFig is True and len(coords)==1):
        plt.figure(figsize=(12,4))
        for s in final_era5["id"].unique()[:1]:
            df = final_era5[final_era5["id"]==s]
            plt.plot(df["date"], df["Tmin"], label="Tmin", c='c')
            plt.plot(df["date"], df["Tavg"], label="Tavg", c='purple')
            plt.plot(df["date"], df["Tmax"], label="Tmax", c='orange')
            df["avg_tmax"] = df["Tmax"].rolling(window=15, center=True).mean()
            plt.plot(df["date"], df["avg_tmax"], label="Tmax rolling", c='r')
            plt.plot(df["date"], df["Prec"], label="Prec", c='b')
        #
        plt.title(f"{agg}")
        plt.ylabel("Weather")
        plt.xticks(rotation=90)
        if (len(coords)==1):
            plt.legend()
        plt.show()

    return final_era5


# --------------------------
def getNDVI(init_dates=None, coords=None, sensor='S2', buffer=1, cloud_probability=None, scale=None, dispFig=False):
    ''' Extract NDVI from different sensors
        
        Parameters:
            init_dates (array): Array of arrays of initial dates. Values in string format 'YYYY-MM-DD'. eg. ['1972-03-13']
            coords (array): Array of arrays of geographic coordinates in longitude, latitude. ie. [[83.45,27.5]]
            sensor (str): Name of the satellite to use in Google Earth Engine (GEE). Default value is 'S2'
            buffer (int): Buffer in meters to extract pixels values around the ROI
            cloud_probability (float): Cloud probability in percentage used with Sentinel-2 (S2).
            scale (int): Size of the pixel in meters used in GEE
            dispFig (bool): Display a figure with the results. 
            
        Usage:
            ``` python
                sowing_date = '2019-01-28'
                lng, lat = 83.45, 27.5
                final_ndvi_S2 = getNDVI(init_dates=[sowing_date], coords=[[lng,lat]], sensor='S2', buffer=buffer, 
                                    cloud_probability=cloud_probability, dispFig=True)
            ```
        
        Returns:
            ndvi (array): A table or pandas dataframe with all NDVI values extracted from the selected sensor or satellite 
        
    '''
    if (sensor not in ['S2', 'MODIS', 'LANDSAT']):
        print("Sensor not valid")
        return
    if (init_dates is None):
        print("Initial dates are not valid")
        return
    if (coords is None):        
        print("Coordinates not valid")
        return
    
    # Init variables
    assert len(coords)==len(init_dates)
    CLOUD_PROBABILITY = cloud_probability
    FEATURES = []
    BUFFER = buffer # meter
    SCALE = 30 # Output resolution in meters.
    if (scale is not None):
        SCALE = scale
    elif (sensor=='S2'):
        SCALE = 10
    elif (sensor=='LANDSAT'):
        SCALE = 30
    elif (sensor=='MODIS'):
        SCALE = 250
    
    # Pre-compute a geographic coordinate system.
    proj = ee.Projection('EPSG:4326').atScale(SCALE).getInfo()
    # Get scales in degrees out of the transform.
    SCALE_X = proj['transform'][0]
    SCALE_Y = -proj['transform'][4]

    if (sensor=='S2'):
        # Blue, green, red, NIR
        FEATURES = ['B2', 'B3', 'B4', 'B8']
    elif (sensor=='LANDSAT'):
        FEATURES = ['B2', 'B3', 'B4', 'B5']
    elif (sensor=='MODIS'):
        FEATURES = ['NDVI']
    else:
        FEATURES = []
        print("None of the bands were selected")
        return
    
    # Applies scaling factors.
    def apply_scale_factors(image):
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True).addBands(
          thermal_bands, None, True )

    # Estimate NDVI
    def calc_NDVI(nir, red): 
        np_ndvis = (nir.astype(float) - red.astype(float)) / (nir + red)
        return np_ndvis

    def get_s2_composite(roi, date, cloudperc=30):
        """Get a two-month Sentinel-2 median composite in the ROI."""
        start = date.advance(-1, 'month')
        end = date.advance(1, 'year')

        s2 = ee.ImageCollection('COPERNICUS/S2')
        s2h = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        s2c = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

        s2c = s2c.filterBounds(roi).filterDate(start, end)
        s2 = s2.filterDate(start, end).filterBounds(roi)

        def indexJoin(collectionA, collectionB, propertyName):
            joined = ee.ImageCollection(ee.Join.saveFirst(propertyName).apply(
                primary=collectionA,
                secondary=collectionB,
                condition=ee.Filter.equals(
                    leftField='system:index',
                    rightField='system:index'
                ))
            )
            return joined.map(lambda image : image.addBands(ee.Image(image.get(propertyName))))

        def maskImage(image):
            s2c = image.select('probability')
            return image.updateMask(s2c.lt(cloudperc))

        withCloudProbability = indexJoin(s2, s2c, 'cloud_probability')
        masked = ee.ImageCollection(withCloudProbability.map(maskImage))
        return masked #.reduce(ee.Reducer.median(), 8)

    @retry.Retry()
    def get_patch(coords, asset_id, bands, buf=1):
        """Get a patch of pixels from an asset, centered on the coords."""
        point = ee.Geometry.Point(coords)
        request = {
            'fileFormat': 'NPY',
            'bandIds': bands,
            'region': point.buffer(buf).bounds().getInfo(), # Get one meter around by default
            'assetId': asset_id
        }
        return np.load(io.BytesIO(ee.data.getPixels(request))) #[band] return all bands
    
    # Creamos funciones para procesar muchos puntos en paralelo 
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=200)
    
    def setup_collection(sowing_date, region, sensor, bands=[], cloud_probability=None):
        sowing = ee.Date(sowing_date)
        start = sowing.advance(-1, 'month')
        end = sowing.advance(1, 'year')
        if (sensor=='S2'):
            col = 'COPERNICUS/S2'
            #s2h = 'COPERNICUS/S2_HARMONIZED'
            #s2c = 'COPERNICUS/S2_CLOUD_PROBABILITY'
            #s2Sr = 'COPERNICUS/S2_SR_HARMONIZED'
        elif (sensor=='LANDSAT'):
            col = 'LANDSAT/LC08/C02/T1_TOA'
        elif (sensor=='MODIS'):
            col = 'MODIS/061/MOD13Q1'
        else:
            print("Collection not found")
            return None
        if (sensor=='S2' and cloud_probability is not None):
            imgCol = get_s2_composite(roi=region, date=sowing, cloudperc=cloud_probability)
        else:
            imgCol = ee.ImageCollection(col).select(bands)
        filter_imgCol = imgCol.filterBounds(region).filterDate(start, end)
        return filter_imgCol.getInfo()['features']
    
    # Process sites by sowind date and coordinates
    final_arrays = []
    for i, (point, sowing_date) in tqdm(enumerate(zip(coords, init_dates)), total=len(coords)):
        #print(i, sd, point)
        #sowing_date = init_dates[i], point = coords[i]
        region = ee.Geometry.Point(point) # Region of interest.
        fts = setup_collection(sowing_date, region, sensor, FEATURES, CLOUD_PROBABILITY)
        future_to_image = {
            executor.submit(get_patch, point, image['id'], FEATURES, BUFFER):
                image['id'] for image in fts
        }
        
        arrays = ()
        types = []
        for future in concurrent.futures.as_completed(future_to_image):
            image_id = future_to_image[future]
            image_name = image_id.split('/')[-1]
            try:
                np_array = future.result()
                # Calculate NDVI
                if (sensor=='S2'):
                    np_ndvi = calc_NDVI(np_array['B8'], np_array['B4'])
                elif (sensor=='LANDSAT'):
                    np_ndvi = calc_NDVI(np_array['B5'], np_array['B4'])
                elif (sensor=='MODIS'):
                    np_ndvi = np_array
                arrays += (np_ndvi,)
                types.append((image_name, np.float_, np_array.shape))
            except Exception as e:
                print(e)
                pass
        final_array = np.array([arrays], types)
        final_arrays.append({"id":i+1, "values":final_array})
    
    # Extract NDVI values by date
    ndvi_values = {}
    for fa in final_arrays:
        _id = fa['id']
        #pnt_name = fa['pnt_name']
        final_values = fa['values']
        for f in final_values.dtype.names:
            if (sensor=='S2'):
                fn = str(f).split('_')
                s2imgdate = fn[0]
                d = dt.datetime.strptime(s2imgdate.split('T')[0], "%Y%m%d")
                #ndvi_vals = round(np.array(final_values[f]).flatten()[0],3)
                ndvi_vals = [ v.mean() for v in final_values[f] ][0].round(3)
            elif (sensor=='LANDSAT'):
                fn = f.split('_')[-1] #'LC08_142041_20190619'
                d = dt.datetime.strptime(fn, "%Y%m%d")
                ndvi_vals = [ v.mean() for v in final_values[f] ][0].round(3) # NDVI scale
            elif (sensor=='MODIS'):
                d = dt.datetime.strptime(f, "%Y_%m_%d")
                ndvi_vals = [ v.mean()*0.0001 for v in final_values[f] ][0].round(3) # NDVI scale

            img_d = d.strftime('%Y-%m-%d')
            year = int(d.strftime('%Y'))
            month = str(d.strftime('%b'))
            doy = int(d.strftime('%j'))
            ndvi_values[f"{_id}_{f}"] = {"id":_id,"image_id":f,"date":img_d, "year":year, "month":month, "doy":doy, "ndvi": ndvi_vals }

    #ndvi_values
    final_ndvi = pd.DataFrame(ndvi_values).T.reset_index(drop=True)
    final_ndvi["ndvi"] = final_ndvi["ndvi"].astype(float)
    final_ndvi["year"] = final_ndvi["year"].astype(int)
    final_ndvi["doy"] = final_ndvi["doy"].astype(int)
    final_ndvi["date"] = pd.to_datetime(final_ndvi["date"].astype(str), format='%Y-%m-%d')
    final_ndvi = final_ndvi.sort_values(["id","date"])
    
    if (dispFig is True): # and len(coords)==1):
        plt.figure(figsize=(12,4))
        for s in final_ndvi["id"].unique():
            df = final_ndvi[final_ndvi["id"]==s]
            if (sensor=='S2'):
                plt.plot(df["date"], df["ndvi"]) #, label="S2 NDVI", c='g')
            elif (sensor=='LANDSAT'):
                plt.plot(df["date"], df["ndvi"]) #, label="LandSat NDVI", c='g')
            elif (sensor=='MODIS'):
                if (len(coords)==1):
                    df["avg_ndvi"] = df["ndvi"].rolling(window=5, center=True).mean()
                    plt.plot(df["date"], df["ndvi"], label="MODIS NDVI", c='g')
                    plt.plot(df["date"], df["avg_ndvi"], label="MODIS NDVI rolling", c='r')
                else:
                    plt.plot(df["date"], df["ndvi"])
        #
        plt.title(f"{sensor}")
        plt.ylabel("NDVI")
        plt.xticks(rotation=90)
        if (len(coords)==1):
            plt.legend()
        plt.show()

    return final_ndvi

# ----------------------