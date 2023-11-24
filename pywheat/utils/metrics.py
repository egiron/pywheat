# coding=utf-8

#******************************************************************************
#
# Metrics
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

import os, sys, gc
import pathlib
import datetime as dt
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score, mean_squared_error, r2_score

def getScores(df, fld1=None, fld2=None):
    ''' Get stats for model results 

        Parameters:
            df (array): A pandas dataframe with Observed and Simulated values 
            fld1 (str): Name of the columns or field with observed values
            fld2 (str): Name of the columns or field with predicted values

        Returns:
            r2score (float): R squared metric
            mape (float): Mean absolute percentage error
            rmse (float): Root mean squared error
            n_rmse (float): Normalized RMSE
            d_index (float): d-index metric
            ef (float): Nash-Sutcliffe metric
            ccc (float): Concordance correlation coefficient
            cb (float): A bias correction factor
            accuracy (float): Accuracy in percentage
        
    '''
    if (df is None):
        print("Input data not valid")
        return
    if (fld1 is None or fld2 is None):
        print("Variable are not valid")
        return
    df_notnull = df[[fld1, fld2]].dropna()
    y_test = df_notnull[fld1].astype('double') #float16
    y_predicted = df_notnull[fld2].astype('double') #float16
    accuracy = round(getAccuracy(y_test, y_predicted),2)
    r2score = round(r2_score(y_test.values, y_predicted.values), 2)
    #r2 = round((np.corrcoef(y_test.values,y_predicted.values)[0, 1])**2, 2)
    rmse = round(mean_squared_error(y_test.values, y_predicted.values, squared=False),2)
    #rmse2 = np.sqrt(np.mean((y_test.values - y_predicted.values)**2))
    #n_rmse = round((rmse / y_test.values.mean()) * 100,2)
    n_rmse = round((rmse / y_test.values.mean()), 3)
    #rmsre = 100 * np.sqrt(np.mean(((y_test.values - y_test.values) / y_test.values)**2))
    mape = round(np.mean(np.abs((y_test.values - y_predicted.values)/y_test.values))*100, 2)
    d1 = ((y_test.values - y_predicted.values).astype('double') ** 2).sum()
    d2 = ((np.abs(y_predicted.values - y_test.values.mean()) + np.abs(y_test.values - y_test.values.mean())).astype('double') ** 2).sum()
    d_index = round(1 - (d1 / d2) ,3)
    # Nash–Sutcliffe model efficiency (EF)
    ef = round(1 - ( np.sum((y_test.values - y_predicted.values)**2) / np.sum((y_test.values - np.mean(y_test.values))**2) ), 2)
    # Concordance correlation coefficient
    #ccc = round(CCC(y_test.values, y_predicted.values),2)
    # A bias correction factor
    #cb = round(Cb(y_test.values, y_predicted.values),2)
    return r2score, mape, rmse, n_rmse, d_index, ef, accuracy #ccc, cb, 

'''
    Calculate accuracy

    Parameters:
        y_true (array): Array of observed values
        y_predicted (array): Array of predicted values
    
    Returns:
        (float): Accuracy in percentage
'''
def getAccuracy(y_true, y_predicted):
    mape = np.mean(np.abs((y_true - y_predicted)/y_true))*100
    if (mape<=100):
        accuracy = np.round((100 - mape), 2)
    else:
        mape = np.mean(np.abs((y_predicted - y_true)/ y_predicted))*100
        accuracy = np.round((100 - mape), 2)
    return accuracy


