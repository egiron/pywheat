# coding=utf-8
#******************************************************************************
#
# Estimating Wheat phenological stages - updated version
# 
# version: 0.0.8
# Copyright: (c) November 2023
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
import time
#import pickle
import numba
import numpy as np
import pandas as pd
import datetime as dt
#from tqdm.notebook import tqdm
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

from numba import njit, jit, prange #cuda, objmode
from numba import int32, int64, float32, float64
from numba.core import types
from numba.typed import Dict
from numba.types import List #, string, optional, UnicodeCharSeq, Tuple

#from ..utils import drawPhenology
from ..utils.metrics import getScores
#from ..data import load_configfiles

@njit(cache=True)
def create_nbDict(keys, values):
    d = Dict()
    assert len(keys) == len(values)
    for idx in range(len(keys)):
        unsigned_idx = np.uint64(idx)
        k = keys[unsigned_idx]
        v = values[unsigned_idx]
        d[k] = v
    return d

def nbdict_populate(pydict):
    keys = list(pydict.keys())
    values = list(pydict.values())
    return create_nbDict(keys, values)

# Overloads
@njit(cache=True)
def numba_to_str2(x):
    with objmode(res=numba.types.unicode_type):
        res = str(round(x,1))
    return res

@njit(cache=True)
def numba_to_str(x):
    return str(x)

@njit(cache=True)
def str_to_posint(s):
	final_index, result = len(s) - 1, 0
	for i,v in enumerate(s):
		result += (ord(v) - 48) * (10 ** (final_index - i))
	return result

@njit(cache=True)
def str_to_int(s):
	neg = (s[0] == "-")
	if(neg): s = s[1:]
	result = str_to_posint(s)
	return -result if neg else result

@njit(cache=True)
def str_to_float(s):
	neg = (s[0] == "-")
	if(neg): s = s[1:]

	exp,dec_loc,exp_loc = 1,-1, len(s)
	for i,c in enumerate(s):
		if(c == "."): 
			dec_loc=i
		elif(c == "e"):
			exp_loc=i
			exp = str_to_int(s[exp_loc+1:])

	if(dec_loc != -1):
		result = str_to_posint(s[:dec_loc])
		result += float(str_to_posint(s[dec_loc+1:exp_loc])) / np.power(10,(exp_loc-dec_loc-1))
	else:
		result = str_to_posint(s)

	if(exp != 1):
		result *= 10.**exp

	if(neg): result = -result
	return result



@njit('int32(int32, int32, float32, float32, float32, float32)', 
      cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True, forceobj=False)
def get_GDD_v2(Tn, Tx, snow_depth=0, Tbase=0, Topt=26, Ttop=34):
    '''
        The daily thermal time (daily_TT) or Growing degree days calculation

        It's calculated from the daily average of maximum and minimum crown temperatures, 
        and is adjusted by genetic and environments factors.

        Parameters:
            snow_depth (int): Snow depth in centimeters (cm). Default value is set to zero.
            Tn (float): Minimum Temperature (°C) * 100
            Tx (float): Maximum Temperature (°C) * 100
            Tbase (float): Base temperature for development from ecotype database. Default 0°C
            Topt (float): Optimum temperature for development from species database. Default 26°C
            Ttop (float): Maximum temperature for development from species database. Default 34°C

        Returns:
            dTT (float): Thermal time or Growing degree days
        
    '''
    _Tn = (Tn / 100)
    _Tx = (Tx / 100)
    Tcmin = 2.0 + _Tn * (0.4 + 0.0018 * ( min(snow_depth, 15) - 15)**2 ) if (_Tn < 0.0) else _Tn
    Tcmax = 2.0 + _Tx * (0.4 + 0.0018 * ( min(snow_depth, 15) - 15)**2 ) if (_Tx < 0.0) else _Tx
    Tcrown = (Tcmax + Tcmin) / 2.0
    tcdif = Tcmax - Tcmin
    dTT = Tcrown - Tbase
    if (tcdif == 0.0): tcdif = 1.0
    if (Tcmax < Tbase):
        dTT = 0.0
    elif(Tcmax < Topt):
        if (Tcmin < Tbase):
            tcor = (Tcmax - Tbase) / tcdif
            dTT = (Tcmax - Tbase) / 2.0 * tcor
        else:
            dTT = Tcrown - Tbase
    elif(Tcmax < Ttop):
        if (Tcmin < Topt):
            tcor = (Tcmax - Topt) / tcdif
            dTT = (Topt - Tbase) / 2.0 * (1.0 + tcor) + Tcmin/2.0 * (1.0 - tcor)
        else:
            dTT = Topt - Tbase
    else:
        if (Tcmin < Topt):
            tcor = (Tcmax - Ttop) / tcdif
            dTT = (Topt + Ttop - Tcmax) * tcor + Topt * (1.0 - tcor)
            tcor =  (Topt - Tcmin) / tcdif
            dTT = dTT * (1 - tcor) + (Tcmin + Topt) / 2.0 * tcor
        else:
            tcor = (Tcmax - Ttop) / tcdif
            dTT = (Topt + Ttop - Tcmax) * tcor + Topt * (1.0 - tcor)
    #
    return round(dTT, 2)

@njit('int32[:](int32[:], int32[:], int32[:], int32, int32, float32, float32, float32, float32, int8, float32, float32, float32, float32 )',
      cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def apply_GDD(Tn, Tx, DOY, sumDTT, maxDTT, snow_depth=0, Tbase=0, Topt=26, Ttop=34, Vern=0, 
              P1V=1.0, P1D=3.675, VREQ=505.0, lat=0.0):
    assert len(Tn)==len(Tx)
    DTT = np.zeros(len(Tn), dtype=np.int32)
    k = 0
    CUMVD = 0
    TDU = 0
    DF = 0.001
    VF = 1.0
    for i in prange(len(Tn)):
        DTT[i] = get_GDD_v2(Tn[i], Tx[i], snow_depth=snow_depth, Tbase=Tbase, Topt=Topt, Ttop=Ttop)
        #DTT += get_GDD_v2(Tn[i], Tx[i], snow_depth=0, Tbase=0, Topt=26, Ttop=34)
        k += 1
        if (Vern==1):
            #Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
            _Tn = (Tn[i] / 100)
            _Tx = (Tx[i] / 100)
            Tcmin = 2.0 + _Tn * (0.4 + 0.0018 * ( min(snow_depth, 15) - 15)**2 ) if (_Tn < 0.0) else _Tn
            Tcmax = 2.0 + _Tx * (0.4 + 0.0018 * ( min(snow_depth, 15) - 15)**2 ) if (_Tx < 0.0) else _Tx
            Tcrown = (Tcmax + Tcmin) / 2.0
            #CUMVD = vernalization(Tcrown, Tn, Tx, CUMVD)
            # Vernalization
            if( (_Tn < 15) and (_Tx > 0.0) ):
                vd1 =  1.4 - 0.0778 * Tcrown
                vd2 =  0.5 + 13.44 / (_Tx - _Tn + 3)**2 * Tcrown # Extract by CERES Wheat 2.0 fortran code
                vd =  min(1.0, vd1, vd2)
                vd =  max(vd, 0.0)
                CUMVD += vd
            elif(_Tx > 30 and CUMVD < 10): # Devernalization
                CUMVD = CUMVD - 0.5 * (_Tx - 30)
                CUMVD = max(CUMVD, 0.0) 
            
            if (CUMVD < VREQ): #params['VREQ']):
                #VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                VF = 1 - (0.0054545 * P1V  + 0.0003) * ( 50 - CUMVD )
                VF = max(min(VF, 1.0), 0.0)
                if (VF < 0.3):
                    TDU = TDU + DTT[i] * min(VF, DF)
                else:
                    # TODOs: 
                    #DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                    #TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                    S1 = np.sin(lat * 0.0174533)
                    C1 = np.cos(lat * 0.0174533)
                    DEC = 0.4093 * np.sin( 0.0172 * (DOY[i] - 82.2) )
                    DLV = ( ( -S1 * np.sin(DEC) - 0.1047 ) / ( C1 * np.cos(DEC) ) )
                    DLV = max(DLV,-0.87)
                    TWILEN = 7.639 * np.arccos(DLV)
                    #DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                    DF = 1 - (0.002 * P1D) * (20 - TWILEN)**2
                    TDU = TDU + DTT[i] * min(VF, DF)
                DTT[i] = TDU
            else:
                Vern=0
        #
        if ((DTT.sum() > sumDTT) or (DTT.sum() >= maxDTT)):
            break
    return DTT[:k].copy()


# ----------------------------

@njit('((int64, types.DictType(unicode_type, float32), types.List(unicode_type, reflected=True), int32, unicode_type, int32, int32, int32, int32, float32, float32, int32[:], int32[:], int32[:], ListType(unicode_type)) )',
      cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def run_pheno_model(i, params, wdates, location, sowing_date, ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, 
                    ObsMaturityDAP, lat, lon, DOYs, Tn, Tx, RESULTS):
    #print(params['TT_TBASE'], WDATES[i][0:5])
    #wdates = WDATES[i]
    SITE = location
    SOWING_DATES = sowing_date 
    GERMINATION_DATES = ''
    EMERGENCE_DATES = '' 
    END_JUVENILE_DATES = ''
    END_VEGETATION_DATES = '' 
    END_OF_EAR_GROWTH_DATES = '' 
    ANTHESIS_DATES = ''
    END_OF_PANNICLE_GROWTH_DATES = '' 
    BEGIN_GRAIN_FILLING_DATES = ''
    END_GRAIN_FILLING_DATES = ''
    HARVEST_DATES = ''

    if (len(Tn)!=len(Tx)):
        return RESULTS
    DAP = 0 # Days after planting
    ISTAGE = 7
    # *******************************
    # DETERMINE PHENOLOGICAL PHASES
    # *******************************
    RESULTS.append(str(i))
    RESULTS.append(str(location))
    # model params
    DAYS_GERMIMATION_LIMIT = int(params['DAYS_GERMIMATION_LIMIT'])
    SNOW=params['SNOW']
    TBASE=params['TT_TBASE']
    TT_TEMPERATURE_OPTIMUM=params['TT_TEMPERATURE_OPTIMUM']
    TT_TEMPERATURE_MAXIMUM=params['TT_TEMPERATURE_MAXIMUM']
    TT_EMERGENCE_LIMIT = params['TT_EMERGENCE_LIMIT']
    GDDE = params['GDDE']
    SDEPTH = params['SDEPTH']
    P1V= params['P1V']
    P1D= params['P1D']
    VREQ= params['VREQ']
    PHINT = params['PHINT']
    TT_TDU_LIMIT = params['TT_TDU_LIMIT']
    # --------------------------------------------------------------------------
    # DETERMINE SOWING DATE
    # --------------------------------------------------------------------------
    if (ISTAGE == 7):
        SOWING_DATES = sowing_date
        CROP_AGE = 0
        # store results
        RESULTS.append(SOWING_DATES)
        #RESULTS.append(str(CROP_AGE))
        ISTAGE = 8
    # --------------------------------------------------------------------------
    # DETERMINE GERMINATION DATE
    # --------------------------------------------------------------------------
    if (ISTAGE == 8):
        SUMDTT = 0
        maxDTT = DAYS_GERMIMATION_LIMIT
        DAP += 1 # Seed germination is a rapid process and is assumed to occur in one day
        _wdates = wdates[0:DAP+1]
        _doy = DOYs[0:DAP+1]
        _Tn = Tn[0:DAP+1]
        _Tx = Tx[0:DAP+1]
        _dTT = apply_GDD(_Tn, _Tx, _doy, DAYS_GERMIMATION_LIMIT, maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=0, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        CROP_AGE = 1
        try:
            if (CROP_AGE < len(_wdates)):
                GERMINATION_DATES = _wdates[CROP_AGE]
                GERMINATION_DOYS = _doy[CROP_AGE]
                GERMINATION_DAPS = DAP
                GERMINATION_CROP_AGE = CROP_AGE
                GERMINATION_SUMDTT = int(SUMDTT)
                # Store results
                RESULTS.append(GERMINATION_DATES)
                RESULTS.append(str(GERMINATION_DOYS))
                RESULTS.append(str(GERMINATION_DAPS))
                RESULTS.append(str(GERMINATION_CROP_AGE))
                RESULTS.append(str(GERMINATION_SUMDTT))
        except:
            # Store results
            RESULTS.append('')
            RESULTS.append('')
            RESULTS.append('')
            RESULTS.append('')
            RESULTS.append('')
        ISTAGE = 9
    # --------------------------------------------------------------------------
    # DETERMINE EMERGENCE DATE
    # --------------------------------------------------------------------------
    if (ISTAGE == 9):
        P9 = 40 + GDDE * SDEPTH
        maxDTT = TT_EMERGENCE_LIMIT
        SUMDTT = 0
        DAP = 0
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        _dTT = apply_GDD(_Tn, _Tx, _doy, P9, maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=0, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT > P9):
            DAP += len(_dTT)
            DAP_9 = len(_dTT)
            CROP_AGE_9 = len(_dTT)
            try:
                if (CROP_AGE_9 < len(_wdates)):
                    EMERGENCE_DATES = _wdates[CROP_AGE_9]
                    EMERGENCE_DOYS = _doy[CROP_AGE_9]
                    EMERGENCE_DAPS = DAP
                    EMERGENCE_CROP_AGE = CROP_AGE_9
                    EMERGENCE_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(EMERGENCE_DATES)
                    RESULTS.append(str(EMERGENCE_DOYS))
                    RESULTS.append(str(EMERGENCE_DAPS))
                    RESULTS.append(str(EMERGENCE_CROP_AGE))
                    RESULTS.append(str(EMERGENCE_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            #
            ISTAGE = 1
    #
    # --------------------------------------------------------------------------------------
    # DETERMINE DURATION OF VEGETATIVE PHASE (END JUVENILE DATE - END OF VEGETATION GROWTH
    # --------------------------------------------------------------------------------------
    if (ISTAGE == 1):
        maxDTT = SUMDTT - P9 + 100
        SUMDTT = SUMDTT - P9 
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        _dTT = apply_GDD(_Tn, _Tx, _doy, P9,  maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=1, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT > P9):
            DAP += len(_dTT)
            DAP_1 = len(_dTT)
            CROP_AGE_1 = len(_dTT)
            try:
                if (CROP_AGE_1 < len(_wdates)):
                    END_JUVENILE_DATES = _wdates[CROP_AGE_1]
                    END_JUVENILE_DOYS = _doy[CROP_AGE_1]
                    END_JUVENILE_DAPS = DAP
                    END_JUVENILE_CROP_AGE = CROP_AGE_1
                    END_JUVENILE_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(END_JUVENILE_DATES)
                    RESULTS.append(str(END_JUVENILE_DOYS))
                    RESULTS.append(str(END_JUVENILE_DAPS))
                    RESULTS.append(str(END_JUVENILE_CROP_AGE))
                    RESULTS.append(str(END_JUVENILE_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            ISTAGE = 1
    # --------------------------------------------------------------------------
    # DETERMINE END VEGETATION DATE - End of Juvenile to End of Vegetative growth
    # --------------------------------------------------------------------------
    #ISTAGE = 1 # <- Note: this must continue with 1 as previous stage (Term Spklt = Emergence to End of Juvenile + End of Juvenile to End of Vegetative growth)
    if (ISTAGE == 1):
        P1 = (TT_TDU_LIMIT * (PHINT / 95.0)) # 400 degree days, Stage 1 development ends
        maxDTT = P1 + 100 #params['TT_TDU_LIMIT']
        VF = 1.0
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        _dTT = apply_GDD(_Tn, _Tx, _doy, P1,  maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=1, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT > P1 ):
            DAP += len(_dTT) + DAP_1 - DAP_9
            CROP_AGE = len(_dTT) + CROP_AGE_1 - CROP_AGE_9 # it should be End Veg - Emergence
            try:
                if (CROP_AGE < len(_wdates)):
                    END_VEGETATION_DATES = _wdates[CROP_AGE]
                    END_VEGETATION_DOYS = _doy[CROP_AGE]
                    END_VEGETATION_DAPS = DAP
                    END_VEGETATION_CROP_AGE = CROP_AGE + CROP_AGE_1
                    END_VEGETATION_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(END_VEGETATION_DATES)
                    RESULTS.append(str(END_VEGETATION_DOYS))
                    RESULTS.append(str(END_VEGETATION_DAPS))
                    RESULTS.append(str(END_VEGETATION_CROP_AGE))
                    RESULTS.append(str(END_VEGETATION_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            ISTAGE = 2
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF EAR GROWTH - End of Vegetative Growth to End of Ear Grow (End leaf growth)
    #-----------------------------------------------------------------------------------------------
    #ISTAGE = 2 # Terminal spikelet initiation to the end of leaf growth - CERES Stage 2
    if (ISTAGE == 2):
        SUMDTT = 0
        P2 = PHINT * 3
        maxDTT = P2 + 100
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        # TODO: Add SNOW from params to this function
        _dTT = apply_GDD(_Tn, _Tx, _doy, P2,  maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=0, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT >= P2):
            DAP += len(_dTT)
            CROP_AGE = len(_dTT)
            try:
                if (CROP_AGE < len(_wdates)):
                    END_OF_EAR_GROWTH_DATES = _wdates[CROP_AGE]
                    HEADING_DOYS = _doy[CROP_AGE]
                    HEADING_DAPS = DAP
                    HEADING_CROP_AGE = CROP_AGE
                    HEADING_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(END_OF_EAR_GROWTH_DATES)
                    RESULTS.append(str(HEADING_DOYS))
                    RESULTS.append(str(HEADING_DAPS))
                    RESULTS.append(str(HEADING_CROP_AGE))
                    RESULTS.append(str(HEADING_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            ISTAGE = 25
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE ANTHESIS
    # ----------------------------------------------------------------------------------------------
    # Anthesis date was estimated as occurring 7 d after heading. (based on McMaster and Smika, 1988; 
    # McMaster and Wilhelm, 2003; G. S. McMaster, unpubl. data). 
    # Here we used 6 days according to IWIN reported anthesis
    #ISTAGE = 2.5
    if (ISTAGE == 25):
        SUMDTT = 0
        ADAH = int(params['ADAH']) # Anthesis days after heading
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:DAP+ADAH]
        _Tx = Tx[DAP:DAP+ADAH]
        _dTT0 = 0
        for jj in range(len(_Tn)):
            _dTT0 += get_GDD_v2(_Tn[jj], _Tx[jj], snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM)
        #
        SUMDTT += _dTT0
        DAP_25 = DAP + ADAH
        CROP_AGE = ADAH
        try:
            if (CROP_AGE < len(_wdates)):
                ANTHESIS_DATES = _wdates[CROP_AGE]
                ANTHESIS_DOYS = _doy[CROP_AGE]
                ANTHESIS_DAPS = DAP_25
                ANTHESIS_CROP_AGE = CROP_AGE
                ANTHESIS_SUMDTT = int(SUMDTT)
                # Store results
                RESULTS.append(ANTHESIS_DATES)
                RESULTS.append(str(ANTHESIS_DOYS))
                RESULTS.append(str(ANTHESIS_DAPS))
                RESULTS.append(str(ANTHESIS_CROP_AGE))
                RESULTS.append(str(ANTHESIS_SUMDTT))
        except:
            # Store results
            RESULTS.append('')
            RESULTS.append('')
            RESULTS.append('')
            RESULTS.append('')
            RESULTS.append('')
        ISTAGE = 3
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF PANNICLE GROWTH - End pannicle growth - End of Ear Growth to Start of Grain Filling
    # ----------------------------------------------------------------------------------------------
    #ISTAGE = 3 # Preanthesis ear growth - CERES Stage 3.
    if (ISTAGE == 3):
        P3 = PHINT * 2
        SUMDTT = 0
        maxDTT = P3 + 100
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        _dTT = apply_GDD(_Tn, _Tx, _doy, P3,  maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=0, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT >= P3):
            DAP += len(_dTT)
            CROP_AGE = len(_dTT)
            try:
                if (CROP_AGE < len(_wdates)):
                    END_OF_PANNICLE_GROWTH_DATES = _wdates[CROP_AGE]
                    END_OF_PANNICLE_GROWTH_DOYS = _doy[CROP_AGE]
                    END_OF_PANNICLE_GROWTH_DAPS = DAP
                    END_OF_PANNICLE_GROWTH_CROP_AGE = CROP_AGE
                    END_OF_PANNICLE_GROWTH_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(END_OF_PANNICLE_GROWTH_DATES)
                    RESULTS.append(str(END_OF_PANNICLE_GROWTH_DOYS))
                    RESULTS.append(str(END_OF_PANNICLE_GROWTH_DAPS))
                    RESULTS.append(str(END_OF_PANNICLE_GROWTH_CROP_AGE))
                    RESULTS.append(str(END_OF_PANNICLE_GROWTH_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            ISTAGE = 4
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE BEGIN GRAIN FILLING - Grain fill - Start of Grain Filling to Maturity
    # ----------------------------------------------------------------------------------------------
    #ISTAGE = 4 # Preanthesis ear growth to the beginning of grain filling - CERES Stage 4.
    if (ISTAGE == 4):
        P4 = params['DSGFT'] #200 GDD # APSIM-Wheat = 120
        maxDTT = P4 + 200
        SUMDTT = 0 #SUMDTT - P3
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        _dTT = apply_GDD(_Tn, _Tx, _doy, P4,  maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=0, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT >= P4):
            DAP += len(_dTT)
            CROP_AGE = len(_dTT)
            try:
                if (CROP_AGE < len(_wdates)):
                    BEGIN_GRAIN_FILLING_DATES = _wdates[CROP_AGE]
                    BEGIN_GRAIN_FILLING_DOYS = _doy[CROP_AGE]
                    BEGIN_GRAIN_FILLING_DAPS = DAP
                    BEGIN_GRAIN_FILLING_CROP_AGE = CROP_AGE
                    BEGIN_GRAIN_FILLING_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(BEGIN_GRAIN_FILLING_DATES)
                    RESULTS.append(str(BEGIN_GRAIN_FILLING_DOYS))
                    RESULTS.append(str(BEGIN_GRAIN_FILLING_DAPS))
                    RESULTS.append(str(BEGIN_GRAIN_FILLING_CROP_AGE))
                    RESULTS.append(str(BEGIN_GRAIN_FILLING_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            ISTAGE = 5
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE END GRAIN FILLING - Maturity
    # ----------------------------------------------------------------------------------------------
    if (ISTAGE == 5):
        #P5 = 430 + params['P5'] * 20 # P5 = (0.05 X TT_Maturity) - 21.5. ~500 degree-days
        P5 = params['P5'] # 400 + 5.0 * 20 
        maxDTT = P5 + 200
        SUMDTT = 0 #SUMDTT - P4
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        _dTT = apply_GDD(_Tn, _Tx, _doy, P5,  maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=0, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT >= P5):
            DAP += len(_dTT)
            CROP_AGE = len(_dTT)
            try:
                if (CROP_AGE < len(_wdates)):
                    #END_GRAIN_FILLING_DATES = _wdates[CROP_AGE]
                    MATURITY_DATES = _wdates[CROP_AGE]
                    MATURITY_DOYS = _doy[CROP_AGE]
                    MATURITY_DAPS = DAP
                    MATURITY_CROP_AGE = CROP_AGE
                    MATURITY_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(MATURITY_DATES)
                    RESULTS.append(str(MATURITY_DOYS))
                    RESULTS.append(str(MATURITY_DAPS))
                    RESULTS.append(str(MATURITY_CROP_AGE))
                    RESULTS.append(str(MATURITY_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            ISTAGE = 6

    # ----------------------------------------------------------------------------------------------
    # DETERMINE HARVEST - Harvest - End of Grain Filling or Maturity
    # ----------------------------------------------------------------------------------------------
    #ISTAGE = 6  # Physiological maturity to harvest - CERES Stage 6.
    if (ISTAGE == 6):
        P6 = params['P6']
        SUMDTT = 0
        maxDTT = P6 + 200
        # A value of 250 degree-days can be used to approximate the thermal time from physiological maturity to harvest
        _wdates = wdates[DAP:]
        _doy = DOYs[DAP:]
        _Tn = Tn[DAP:]
        _Tx = Tx[DAP:]
        _dTT = apply_GDD(_Tn, _Tx, _doy, P6,  maxDTT, snow_depth=SNOW, Tbase=TBASE, 
                         Topt=TT_TEMPERATURE_OPTIMUM, Ttop=TT_TEMPERATURE_MAXIMUM, 
                         Vern=0, P1V=P1V, P1D=P1D, VREQ=VREQ, lat=lat)
        SUMDTT += _dTT.sum()
        if (SUMDTT >= P6):
            DAP += len(_dTT)
            CROP_AGE = len(_dTT)
            try:
                if (CROP_AGE < len(_wdates)):
                    HARVEST_DATES = _wdates[CROP_AGE]
                    HARVEST_DOYS = _doy[CROP_AGE]
                    HARVEST_DAPS = DAP
                    HARVEST_CROP_AGE = CROP_AGE
                    HARVEST_SUMDTT = int(SUMDTT)
                    # Store results
                    RESULTS.append(HARVEST_DATES)
                    RESULTS.append(str(HARVEST_DOYS))
                    RESULTS.append(str(HARVEST_DAPS))
                    RESULTS.append(str(HARVEST_CROP_AGE))
                    RESULTS.append(str(HARVEST_SUMDTT))
            except:
                # Store results
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
                RESULTS.append('')
            ISTAGE = -99
    #
    return RESULTS
    
    
# ----------------------------------
def init_params(initparams=None):
    # Initialization of variables 
    params = dict(
        TT_TBASE = 0.0, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = 26.0, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = 34.0, # Thermal time maximum temperature
        CIVIL_TWILIGHT = 0.0, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        HI = 0.0, # Hardiness Index
        SNOW = 0.0, # Snow fall
        SDEPTH = 3.0, # Sowing depth in cm
        GDDE = 6.2, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = 200.0, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = 505.0, # Vernalization required for max.development rate (VDays)
        PHINT = 95.0, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = 1.0, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = 3.675, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = 500.0, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = 250.0, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = 40.0, # threshold for days to germination
        TT_EMERGENCE_LIMIT = 400.0, # threshold for thermal time to emergence
        TT_TDU_LIMIT = 1000.0, # threshold for thermal development units (TDU)
        ADAH = 6.0, # threshold for anthesis date after planting. This is a 6 days after heading.
    )
    # Update parameters
    if (initparams is not None):
        params = {**params, **initparams}

    # Convert to numba Dict
    try:
        #params = nbdict_populate(params)
        keys = list(params.keys())
        values = list([numba.float32(v) for k,v in params.items()]) # Convert all variables to float to avoid conflict with typed numba dict 
        params = create_nbDict(keys, values)
    except Exception as err:
        print("Problem with the initialization of variables. Error:", err)
    return params

# --------------------------------
# BRUTE FORCE
# --------------------------------
# Getting Emergence by brute force algorithm
@njit(cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def estimate_emergence_by_bruteforce_v3(uid, params, wdates, location, sowing_date, 
                                        ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                        lat, lon, DOYs, Tn, Tx, max_tries=300, gdde_steps=0.05, maxGDDE=50):
    if (params is None):
        print("Parameters not valid")
        return
    if (ObsEmergenceDAP is None):
        print("Observed emergence days after planting not defined")
        return
    status = -1
    try:
        # Setup initial parameters
        e_params = params.copy()
        TT_EMERGENCE_LIMIT = e_params['TT_EMERGENCE_LIMIT']
        SDEPTH = e_params['SDEPTH']
        GDDE = e_params['GDDE']
        # Run initial simulation
        #new_params = dict( SDEPTH=SDEPTH, GDDE=GDDE, TT_EMERGENCE_LIMIT=TT_EMERGENCE_LIMIT )
        #simDAP = run_calibation_pheno_model(new_params, obs_arr, config)['SimEmergenceDAP'].values[0]
        #params = init_params(new_params)
        RESULTS = numba.typed.List.empty_list(numba.types.string)
        _arr2 = run_pheno_model(uid, e_params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, RESULTS)
        try:
            simDAP = str_to_int(_arr2[10]) #e_DAP
            SUMDTT = str_to_int(_arr2[12]) # e_SUMDTT
        except:
            status = -1
            return e_params, status
            
        # loop until converge
        t = 0
        while True:
            if (simDAP < ObsEmergenceDAP):
                GDDE = GDDE + gdde_steps
                if GDDE > 20:
                    SDEPTH = 5.0
                if GDDE > 30:
                    SDEPTH = 10.0
                if GDDE > 40:
                    SDEPTH = 15.0
                if GDDE > 50:
                    SDEPTH = 20.0
                #    SDEPTH = min(SDEPTH, 20.0)
            else:
                GDDE = GDDE - gdde_steps
                GDDE = max(GDDE, 0.0)
                SDEPTH = 3.0

            #new_params = dict( SDEPTH=SDEPTH, GDDE=GDDE ) # Updated parameters
            #params = {**params, **new_params}
            e_params['SDEPTH'] = SDEPTH
            e_params['GDDE'] = GDDE
            #res = run_calibation_pheno_model(params, obs_arr, config)
            #simDAP = res['SimEmergenceDAP'].values[0]
            #SUMDTT = res['SimEmergenceSUMDTT'].values[0]
            
            #e_params = init_params(new_params)
            RESULTS = numba.typed.List.empty_list(numba.types.string)
            _arr3 = run_pheno_model(uid, e_params, wdates, location, sowing_date, 
                                                    ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                    lat, lon, DOYs, Tn, Tx, RESULTS)
            #res_arr3 = [str(x) for x in _arr3] # Covert from numba.List to numpy array and append
            try:
                simDAP = str_to_int(_arr3[10]) #e_DAP
                SUMDTT = str_to_int(_arr3[12]) # e_SUMDTT
                #print("DAP -> ",simDAP, obsDAP)
                #simDAP = int(simDAP) # Problem with DAP = '' # not found
            except:
                #status = -1
                pass
            if (simDAP == ObsEmergenceDAP):
                status = 1
                break
            elif (SUMDTT > TT_EMERGENCE_LIMIT):
                status = -1
                break
            elif GDDE > maxGDDE or GDDE <= 0:
                status = -1
                break
            elif (t > max_tries):
                status = -1
                break
            t += 1

        # end while
    except:
        #print(f"Problem getting emergence by brute force")
        pass
    return e_params, status



# Getting Heading by brute force algorithm
@njit(cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def estimate_heading_by_bruteforce_v3(uid, params, wdates, location, sowing_date, 
                                        ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                        lat, lon, DOYs, Tn, Tx, max_tries=800, phint_steps=0.05, maxPHINT=150):
    if (params is None):
        print("Parameters not valid")
        return
    if (ObsHeadingDAP is None):
        print("Observed heading days after planting not defined")
        return
    status = -2
    try:
        # Setup initial parameters
        h_params = params.copy()
        h_params['TT_TDU_LIMIT'] = 10000 # Value to allows fit the model or the point converges to the abs minimum
        TT_TDU_LIMIT = h_params['TT_TDU_LIMIT']
        SDEPTH = h_params['SDEPTH']
        GDDE = h_params['GDDE']
        SNOW = h_params['SNOW'] #0 
        VREQ = h_params['VREQ'] #505.0 
        PHINT = h_params['PHINT'] #95.0
        P1V = h_params['P1V'] #1.0 # Spring wheat
        P1D = h_params['P1D'] #3.675
        #
        RESULTS = numba.typed.List.empty_list(numba.types.string)
        _arr2 = run_pheno_model(uid, h_params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, RESULTS)
        try:
            simDAP = str_to_int(_arr2[25]) # h_DAP
            SUMDTT = str_to_int(_arr2[27]) # h_SUMDTT
        except:
            status = -2
            return h_params, status
        # loop until converge
        t = 0
        while True:
            if (simDAP < ObsHeadingDAP):
                PHINT += phint_steps
                PHINT = min(PHINT, maxPHINT)
                #if PHINT > 110.0:
                P1V += 0.01
                P1V = min(P1V, 5.0)
                #P1V += 0.05
                #P1D += 0.05
                #P1D = min(P1D, 6.0)
                #VREQ += 0.15
                #VREQ = min(VREQ, 800.0)
                #SNOW += 0.5
                #SNOW = min(SNOW, 20.0)
                #TT_TDU_LIMIT -= 1
                #TT_TDU_LIMIT = max(TT_TDU_LIMIT, 0.0)
            else:
                PHINT -= phint_steps
                PHINT = max(PHINT, 0.0)
                P1V -= 0.05
                P1V = max(P1V, 0.0)
                P1D -= 0.05
                P1D = max(P1D, 0.0)
                VREQ -= 0.15
                VREQ = max(VREQ, 0.0)
                #P1V -= 0.05
                #P1V = max(P1V, 0.0)
                #P1D -= 0.05
                #P1D = max(P1D, 0.0)
                #VREQ -= 0.15
                #VREQ = max(VREQ, 0.0)
                #TT_TDU_LIMIT += 1
                #TT_TDU_LIMIT = min(TT_TDU_LIMIT, 800.0)
            #
            #new_params = dict( 
            #    PHINT=PHINT, P1V=P1V, P1D=P1D, VREQ=VREQ, SNOW=SNOW, TT_TDU_LIMIT=TT_TDU_LIMIT
            #)
            #params = {**params, **new_params}
            h_params['SNOW'] = SNOW
            h_params['VREQ'] = VREQ
            h_params['PHINT'] = PHINT
            h_params['P1V'] = P1V
            h_params['P1D'] = P1D
            #h_params['TT_TDU_LIMIT'] = 10000
            # Run simulation
            RESULTS = numba.typed.List.empty_list(numba.types.string)
            _arr3 = run_pheno_model(uid, h_params, wdates, location, sowing_date, 
                                                    ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                    lat, lon, DOYs, Tn, Tx, RESULTS)
            try:
                simDAP = str_to_int(_arr3[25]) # h_DAP
                SUMDTT = str_to_int(_arr3[27]) # h_SUMDTT
            except:
                #status = -2
                pass
            if (simDAP == ObsHeadingDAP):
                status = 2
                break
            elif (PHINT > maxPHINT or PHINT <= 0):
                status = -2
                break
            elif (t > max_tries):
                status = -2
                break
            t += 1

        # end while

    except: # Exception as err:
        #print(f"Problem getting heading by brute force",err)
        pass

    return h_params, status

# Getting anthesis by brute force algorithm
@njit(cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def estimate_anthesis_by_bruteforce_v3(uid, params, wdates, location, sowing_date, 
                                        ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                        lat, lon, DOYs, Tn, Tx, max_tries=300, adap_steps=1, maxADAP=10):
    if (params is None):
        print("Parameters not valid")
        return
    if (ObsAnthesisDAP is None):
        print("Observed heading days after planting not defined")
        return
    status = -3
    try:
        # Setup initial parameters
        a_params = params.copy()
        ADAH = a_params['ADAH']
        #
        RESULTS = numba.typed.List.empty_list(numba.types.string)
        _arr2 = run_pheno_model(uid, a_params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, RESULTS)
        try:
            simDAP = str_to_int(_arr2[30]) # a_DAP
            SUMDTT = str_to_int(_arr2[32]) # a_SUMDTT
        except:
            status = -3
            return a_params, status
        # loop until converge
        t = 0
        while True:
            if (simDAP < ObsAnthesisDAP):
                ADAH = ADAH + adap_steps
                #ADAH = min(maxADAP, ADAH)
            else:
                ADAH = ADAH - adap_steps
                ADAH = max(ADAH, 0.0)
            #
            a_params['ADAH'] = ADAH
            # Run simulation
            RESULTS = numba.typed.List.empty_list(numba.types.string)
            _arr3 = run_pheno_model(uid, a_params, wdates, location, sowing_date, 
                                                    ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                    lat, lon, DOYs, Tn, Tx, RESULTS)
            try:
                simDAP = str_to_int(_arr3[30]) # a_DAP
                SUMDTT = str_to_int(_arr3[32]) # a_SUMDTT
            except:
                #status = -3
                pass
            if (simDAP == ObsAnthesisDAP):
                status = 3
                break
            elif (ADAH > maxADAP or ADAH <= 0):
                status = -3
                break
            elif (t > max_tries):
                status = -3
                break
            t += 1

        # end while

    except: # Exception as err:
        #print(f"Problem getting heading by brute force",err)
        pass

    return a_params, status

# Getting maturity by brute force algorithm
@njit(cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def estimate_maturity_by_bruteforce_v3(uid, params, wdates, location, sowing_date, 
                                        ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                        lat, lon, DOYs, Tn, Tx, max_tries=500, p5_steps=1.0, maxP5=5000):
    if (params is None):
        print("Parameters not valid")
        return
    if (ObsMaturityDAP is None):
        print("Observed heading days after planting not defined")
        return
    status = -4
    try:
        # Setup initial parameters
        m_params = params.copy()
        P4 = m_params['DSGFT'] #200 GDD # APSIM-Wheat = 120
        P5 = m_params['P5'] # P5 = (0.05 X TT_Maturity) - 21.5. ~500 degree-days 
        
        RESULTS = numba.typed.List.empty_list(numba.types.string)
        _arr2 = run_pheno_model(uid, m_params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, RESULTS)
        try:
            simDAP = str_to_int(_arr2[45]) # m_DAP
            SUMDTT = str_to_int(_arr2[47]) # m_SUMDTT
        except:
            status = -4
            return m_params, status
        # loop until converge
        t = 0
        while True:
            if (simDAP < ObsMaturityDAP):
                P5 = P5 + p5_steps
                #P5 = min(maxP5, P5)
            else:
                P5 = P5 - p5_steps
                P5 = max(P5, 0.0)
                P4 = P4 - 1
                P4 = max(P4, 50.0)
            #
            m_params['DSGFT'] = P4
            m_params['P5'] = P5
            # Run simulation
            RESULTS = numba.typed.List.empty_list(numba.types.string)
            _arr3 = run_pheno_model(uid, m_params, wdates, location, sowing_date, 
                                                    ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                    lat, lon, DOYs, Tn, Tx, RESULTS)
            try:
                simDAP = str_to_int(_arr3[45]) # m_DAP
                SUMDTT = str_to_int(_arr3[47]) # m_SUMDTT
            except:
                #status = -4
                pass
            if (simDAP == ObsMaturityDAP):
                status = 4
                break
            elif (P5 > maxP5 or P5 <= 0):
                status = -4
                break
            elif (t > max_tries):
                status = -4
                break
            t += 1
        # end while
    except: # Exception as err:
        #print(f"Problem getting heading by brute force",err)
        pass

    return m_params, status

# --------------------------------------
# RUN CALIBRATION USING IWIN DATASET
# --------------------------------------
def run_calibration_pheno_model_numba(df_IWIN, arr_final):
    #params3 = init_params(initparams)
    params = init_params()
    
    start_time = time.perf_counter()
    # Run model for each site
    res_arr = []
    params_arr = []
    sz = 400
    idx_data = 34
    
    # Warm up
    for i in range(len(arr_final[:100])):
        UID = int32(arr_final[i][0])
        location = int32(arr_final[i][3]) #location
        sowing_date = arr_final[i][10] #sowing
        lat = float32(arr_final[i][17])
        lon = float32(arr_final[i][18])
        ObsEmergenceDAP = int32(arr_final[i][25])
        ObsHeadingDAP = int32(arr_final[i][26])
        ObsAnthesisDAP = int32(arr_final[i][27])
        ObsMaturityDAP = int32(arr_final[i][28])
        wdates = list(np.asarray(arr_final[i][idx_data:sz*1+idx_data], dtype='<U32'))
        DOYs = np.asarray(arr_final[i][sz*1+idx_data:sz*2+idx_data], dtype='int32')
        Tn = np.asarray(arr_final[i][sz*2+idx_data:sz*3+idx_data], dtype='int32')
        Tx = np.asarray(arr_final[i][sz*3+idx_data:sz*4+idx_data], dtype='int32')
        # Warm up
        R = numba.typed.List.empty_list(numba.types.string)
        _ = run_pheno_model(i, params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, R)
    # Run models
    for i in tqdm(range(len(arr_final))):
        UID = int32(arr_final[i][0])
        location = int32(arr_final[i][3]) #location
        sowing_date = arr_final[i][10] #sowing
        lat = float32(arr_final[i][17])
        lon = float32(arr_final[i][18])
        ObsEmergenceDAP = int32(arr_final[i][25])
        ObsHeadingDAP = int32(arr_final[i][26])
        ObsAnthesisDAP = int32(arr_final[i][27])
        ObsMaturityDAP = int32(arr_final[i][28])
        
        wdates = list(np.asarray(arr_final[i][idx_data:sz*1+idx_data], dtype='<U32'))
        DOYs = np.asarray(arr_final[i][sz*1+idx_data:sz*2+idx_data], dtype='int32')
        Tn = np.asarray(arr_final[i][sz*2+idx_data:sz*3+idx_data], dtype='int32')
        Tx = np.asarray(arr_final[i][sz*3+idx_data:sz*4+idx_data], dtype='int32')
        #rhx = np.asarray(arr_final[i][sz*4+idx_data:sz*5+idx_data], dtype='int32')
        #pcp = np.asarray(arr_final[i][sz*5+idx_data:sz*6+idx_data], dtype='int32')
        #srad = np.asarray(arr_final[i][sz*6+idx_data:sz*7+idx_data], dtype='int32')
        #
        if ((ObsEmergenceDAP != -99) and (ObsEmergenceDAP > 3) and (ObsEmergenceDAP <= 40)):
            e_params, e_status = estimate_emergence_by_bruteforce_v3(UID, params, wdates, location, sowing_date, 
                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                            lat, lon, DOYs, Tn, Tx, max_tries=500, gdde_steps=0.05, maxGDDE=50)
            # HEADING
            if ((ObsHeadingDAP != -99) and (ObsHeadingDAP < 250)):
                h_params, h_status = estimate_heading_by_bruteforce_v3(UID, e_params, wdates, location, sowing_date, 
                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                            lat, lon, DOYs, Tn, Tx, max_tries=800, phint_steps=0.05, maxPHINT=150)
                # Anthesis
                if (ObsAnthesisDAP != -99):
                    a_params, a_status = estimate_anthesis_by_bruteforce_v3(UID, h_params, wdates, location, sowing_date, 
                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                            lat, lon, DOYs, Tn, Tx, max_tries=300, adap_steps=1, maxADAP=10)
                else:
                    a_status = 0
            else:
                h_status = 0
                h_params = e_params.copy()
                a_status = 0
            #
            # MATURITY
            if ((ObsMaturityDAP != -99) and (ObsMaturityDAP < 350)):
                m_params, m_status = estimate_maturity_by_bruteforce_v3(UID, h_params, wdates, location, sowing_date, 
                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                            lat, lon, DOYs, Tn, Tx, max_tries=500, p5_steps=1.0, maxP5=5000)
            else:
                m_params = h_params.copy()
                m_status = 0
            #
            RESULTS = numba.typed.List.empty_list(numba.types.string)
            _arr = run_pheno_model(UID, m_params, wdates, location, sowing_date, 
                                                    ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                    lat, lon, DOYs, Tn, Tx, RESULTS)
            _arr.append(str(e_status))
            _arr.append(str(h_status))
            _arr.append(str(a_status))
            _arr.append(str(m_status))
            # save params
            params_arr.append({'UID':UID, 'params':m_params})
            #_arr.append(str(m_params['TT_EMERGENCE_LIMIT']))
            #_arr.append(str(m_params['TT_TDU_LIMIT']))
            #_arr.append(str(m_params['SDEPTH']))
            #_arr.append(str(m_params['GDDE']))
            #_arr.append(str(m_params['SNOW']))
            #_arr.append(str(m_params['VREQ']))
            #_arr.append(str(m_params['PHINT']))
            #_arr.append(str(m_params['P1V']))
            #_arr.append(str(m_params['P1D']))
            #_arr.append(str(m_params['DSGFT']))
            #_arr.append(str(m_params['P5']))
            #_arr.append(str(m_params['P6']))
            #_arr.append(str(m_params['ADAH']))
        else:
            # HEADING
            if ((ObsHeadingDAP != -99) and (ObsHeadingDAP < 250)):
                h_params, h_status = estimate_heading_by_bruteforce_v3(UID, params, wdates, location, sowing_date, 
                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                            lat, lon, DOYs, Tn, Tx, max_tries=800, phint_steps=0.05, maxPHINT=150)
                #
                if (ObsAnthesisDAP != -99):
                    a_params, a_status = estimate_anthesis_by_bruteforce_v3(UID, h_params, wdates, location, sowing_date, 
                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                            lat, lon, DOYs, Tn, Tx, max_tries=300, adap_steps=1, maxADAP=10)
                else:
                    a_status = 0
    
                # MATURITY
                if ((ObsMaturityDAP != -99) and (ObsMaturityDAP < 350)):
                    m_params, m_status = estimate_maturity_by_bruteforce_v3(UID, h_params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, max_tries=500, p5_steps=1.0, maxP5=5000)
                else:
                    m_params = h_params.copy()
                    m_status = 0
                #
                RESULTS = numba.typed.List.empty_list(numba.types.string)
                _arr = run_pheno_model(UID, m_params, wdates, location, sowing_date, 
                                                        ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                        lat, lon, DOYs, Tn, Tx, RESULTS)
                _arr.append(str(0))
                _arr.append(str(h_status))
                _arr.append(str(a_status))
                _arr.append(str(m_status))
                # save params
                params_arr.append({'UID':UID, 'params':m_params})
                #_arr.append(str(m_params['TT_EMERGENCE_LIMIT']))
                #_arr.append(str(m_params['TT_TDU_LIMIT']))
                #_arr.append(str(m_params['SDEPTH']))
                #_arr.append(str(m_params['GDDE']))
                #_arr.append(str(m_params['SNOW']))
                #_arr.append(str(m_params['VREQ']))
                #_arr.append(str(m_params['PHINT']))
                #_arr.append(str(m_params['P1V']))
                #_arr.append(str(m_params['P1D']))
                #_arr.append(str(m_params['DSGFT']))
                #_arr.append(str(m_params['P5']))
                #_arr.append(str(m_params['P6']))
                #_arr.append(str(m_params['ADAH']))
    
            else:
                # ANTHESIS
                if (ObsAnthesisDAP != -99):
                    a_params, a_status = estimate_anthesis_by_bruteforce_v3(UID, params, wdates, location, sowing_date, 
                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                            lat, lon, DOYs, Tn, Tx, max_tries=300, adap_steps=1, maxADAP=10)
                else:
                    a_status = 0
                # MATURITY
                if ((ObsMaturityDAP != -99) and (ObsMaturityDAP < 350)):
                    m_params, m_status = estimate_maturity_by_bruteforce_v3(UID, params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, max_tries=500, p5_steps=1.0, maxP5=5000)
                    #
                    RESULTS = numba.typed.List.empty_list(numba.types.string)
                    _arr = run_pheno_model(UID, m_params, wdates, location, sowing_date, 
                                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                            lat, lon, DOYs, Tn, Tx, RESULTS)
                    _arr.append(str(0))
                    _arr.append(str(0))
                    _arr.append(str(a_status))
                    _arr.append(str(m_status))
                    # save params
                    params_arr.append({'UID':UID, 'params':m_params})
                    #_arr.append(str(m_params['TT_EMERGENCE_LIMIT']))
                    #_arr.append(str(m_params['TT_TDU_LIMIT']))
                    #_arr.append(str(m_params['SDEPTH']))
                    #_arr.append(str(m_params['GDDE']))
                    #_arr.append(str(m_params['SNOW']))
                    #_arr.append(str(m_params['VREQ']))
                    #_arr.append(str(m_params['PHINT']))
                    #_arr.append(str(m_params['P1V']))
                    #_arr.append(str(m_params['P1D']))
                    #_arr.append(str(m_params['DSGFT']))
                    #_arr.append(str(m_params['P5']))
                    #_arr.append(str(m_params['P6']))
                    #_arr.append(str(m_params['ADAH']))
                else:
                    # Modify some params to reach Harvest and avoid wrong values in the final DF
                    #params['TT_EMERGENCE_LIMIT'] = 500.0
                    #params['TT_TDU_LIMIT'] = 1000.0
                    RESULTS = numba.typed.List.empty_list(numba.types.string)
                    _arr = run_pheno_model(UID, params, wdates, location, sowing_date, 
                                                            ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                            lat, lon, DOYs, Tn, Tx, RESULTS)
                    _arr.append(str(0))
                    _arr.append(str(0))
                    _arr.append(str(0))
                    _arr.append(str(0))
                    # save params
                    params_arr.append({'UID':UID, 'params':params})
                    #_arr.append(str(params['TT_EMERGENCE_LIMIT']))
                    #_arr.append(str(params['TT_TDU_LIMIT']))
                    #_arr.append(str(params['SDEPTH']))
                    #_arr.append(str(params['GDDE']))
                    #_arr.append(str(params['SNOW']))
                    #_arr.append(str(params['VREQ']))
                    #_arr.append(str(params['PHINT']))
                    #_arr.append(str(params['P1V']))
                    #_arr.append(str(params['DSGFT']))
                    #_arr.append(str(params['P1D']))
                    #_arr.append(str(params['P5']))
                    #_arr.append(str(params['P6']))
                    #_arr.append(str(params['ADAH']))
        res_arr.append([str(x) for x in _arr]) # Covert from numba.List to numpy array and append
    elapsed_time = time.perf_counter() - start_time
    print(f"Total: {elapsed_time:.3f} seconds")
    
    #
    if (len(res_arr)>0):
        ph = pd.DataFrame(res_arr)
        ph.columns = ['UID','location','Sowing', 'SimGermination', 'g_DOY', 'g_DAP', 'g_AGE', 'g_SUMDTT',
                      'SimEmergence', 'e_DOY', 'e_DAP', 'e_AGE', 'e_SUMDTT',
                      'Sim Term Spklt', 'ts_DOY', 'ts_DAP', 'ts_AGE', 'ts_SUMDTT',
                      'Sim End Veg', 'ev_DOY', 'ev_DAP', 'ev_AGE', 'ev_SUMDTT',
                      'SimHeading', 'h_DOY', 'h_DAP', 'h_AGE', 'h_SUMDTT', #'End Ear Gr or Heading'
                      'SimAnthesis', 'a_DOY', 'a_DAP', 'a_AGE', 'a_SUMDTT', 
                      'Sim End Pan Gr', 'epg_DOY', 'epg_DAP', 'epg_AGE', 'epg_SUMDTT',
                      'Sim Beg Gr Fil', 'bgf_DOY', 'bgf_DAP', 'bgf_AGE', 'bgf_SUMDTT',
                      'SimMaturity', 'm_DOY', 'm_DAP', 'm_AGE', 'm_SUMDTT', #'End Gr Fil', 
                      'SimHarvest', 'hv_DOY', 'hv_DAP', 'hv_AGE', 'hv_SUMDTT', 
                      'e_status', 'h_status', 'a_status', 'm_status', 
                      #'TT_EMERGENCE_LIMIT', 'TT_TDU_LIMIT',
                      #'SDEPTH', 'GDDE', 'SNOW', 'VREQ', 'PHINT', 'P1V', 'P1D', 'DSGFT', 'P5', 'P6', 'ADAH'
                     ]
        # Try to convert formats
        try:
            dates_columns = ['Sowing', 'SimGermination', 'SimEmergence', 'Sim Term Spklt', 'Sim End Veg', 
                             'SimHeading', #'End Ear Gr or Heading'
                              'SimAnthesis', 'Sim End Pan Gr','Sim Beg Gr Fil', 'SimMaturity', #'End Gr Fil', 
                              'SimHarvest']
            
            ph[dates_columns] = ph[dates_columns].apply(lambda x: pd.to_datetime(x.astype(str), format="%Y-%m-%d"), axis=0 )
            
        except Exception as err:
            pass
        try:
            intcols = ['UID','location']
            ph[intcols] = ph[intcols].astype(np.uint32)
            fcols = ['g_DOY', 'g_DAP', 'g_AGE', 'g_SUMDTT',
                      'e_DOY', 'e_DAP', 'e_AGE', 'e_SUMDTT',
                      'ts_DOY', 'ts_DAP', 'ts_AGE', 'ts_SUMDTT',
                      'ev_DOY', 'ev_DAP', 'ev_AGE', 'ev_SUMDTT',
                      'h_DOY', 'h_DAP', 'h_AGE', 'h_SUMDTT', #'End Ear Gr or Heading'
                      'a_DOY', 'a_DAP', 'a_AGE', 'a_SUMDTT', 
                      'epg_DOY', 'epg_DAP', 'epg_AGE', 'epg_SUMDTT',
                      'bgf_DOY', 'bgf_DAP', 'bgf_AGE', 'bgf_SUMDTT',
                      'm_DOY', 'm_DAP', 'm_AGE', 'm_SUMDTT', #'End Gr Fil', 
                      'hv_DOY', 'hv_DAP', 'hv_AGE', 'hv_SUMDTT', 
                      'e_status', 'h_status', 'a_status', 'm_status',
                      #'TT_EMERGENCE_LIMIT', 'TT_TDU_LIMIT',
                      #'SDEPTH', 'GDDE', 'SNOW', 'VREQ', 'PHINT', 'P1V', 'P1D', 'DSGFT', 'P5', 'P6', 'ADAH'
                    ]
            ph[fcols] = ph[fcols].astype(np.float32)
        except Exception as err:
            pass
        
        try:
            df = pd.merge(df_IWIN[['UID','location', 'lat', 'lon', 'sowing', 'emergence', 'heading', 'maturity',
                          'ObsEmergDOY','ObsEmergDAP', 'ObsHeadingDOY', 'ObsHeadingDAP', 'ObsAnthesisDOY', 'ObsAnthesisDAP',
                          'ObsMaturityDAP', 'ObsMaturityDOY'
                         ]], ph, on=['UID'], how='right',
                )
        
            df['eresidualsDAP'] = df['ObsEmergDAP'] - df['e_DAP']
            df['eresidualsDOY'] = df['ObsEmergDOY'] - df['e_DOY']
            df['hresidualsDAP'] = df['ObsHeadingDAP'] - df['h_DAP']
            df['hresidualsDOY'] = df['ObsHeadingDOY'] - df['h_DOY']
            df['aresidualsDAP'] = df['ObsAnthesisDAP'] - df['a_DAP']
            df['aresidualsDOY'] = df['ObsAnthesisDOY'] - df['a_DOY']
            df['mresidualsDAP'] = df['ObsMaturityDAP'] - df['m_DAP']
            df['mresidualsDOY'] = df['ObsMaturityDOY'] - df['m_DOY']
            #
            df.rename(columns={'location_x': 'location'}, inplace=True)
            df.drop(columns=['location_y', 'Sowing'], inplace=True)
            # Join calibrated params - params_arr
            # Join params with rest of the attributes
            try:
                df_params = pd.DataFrame([params_arr[x]['params'] for x in range(len(params_arr))])
                df_uids = pd.DataFrame([params_arr[x]['UID'] for x in range(len(params_arr))])
                df_iwin_params = pd.concat([df_uids, df_params], axis=1)
                df_iwin_params = df_iwin_params.rename(columns={0:'UID'})
                #hoy = dt.datetime.now().strftime('%Y%m%d')
                #df.to_csv(f'df_iwin_params_calibrated_{hoy}.csv')
                df = pd.merge(df, df_iwin_params, on='UID')
            except Exception as err:
                pass
            hoy = dt.datetime.now().strftime('%Y%m%d')
            df.to_parquet(f'df_iwin_calibration_models_{hoy}.parquet')
            return df
        except Exception as err:
            pass
    #
    return ph

# -------------------------------------------
''' Get calibration parameters '''
def get_calibrated_params(sowing_date=None, latitude=None, longitude=None, params=None, config=None):
    if (sowing_date is None or latitude is None or longitude is None or config is None 
        or params is None):
        return params
    # Extract values from configuration file "configbycoords" using K-Nearest neighbors
    sowing_date_ms = time.mktime(dt.datetime.strptime(sowing_date, "%Y-%m-%d").timetuple())*1000
    new_site = pd.DataFrame({"lat": [latitude], "lon": [longitude], "sowing_date": [sowing_date_ms] }) 
    nearest = config["nn"].kneighbors(new_site, n_neighbors=3, return_distance=False)
    row = config["configbycoords"].iloc[nearest[0]]
    # 'lat', 'lon', 'sowing_date', 'smonth', 'SDEPTH', 'GDDE', 'SNOW', 'VREQ',
    # 'PHINT', 'P1V', 'P1D', 'P5', 'P6', 'DSGFT', 'ADAH', 'e_DAP', 'h_DAP',
    # 'a_DAP', 'm_DAP', 'e_SUMDTT', 'h_SUMDTT', 'a_SUMDTT', 'm_SUMDTT'
    new_params = params.copy()
    new_params['SDEPTH'] = float32(row['SDEPTH'].mean())
    new_params['GDDE'] = float32(row['GDDE'].mean())
    new_params['SNOW'] = float32(row['SNOW'].mean())
    new_params['VREQ'] = float32(row['VREQ'].mean())
    new_params['PHINT'] = float32(row['PHINT'].mean())
    new_params['P1V'] = float32(row['P1V'].mean())
    new_params['P1D'] = float32(row['P1D'].mean())
    new_params['P5'] = float32(row['P5'].mean())
    new_params['P6'] = float32(row['P6'].mean())
    new_params['DSGFT'] = float32(row['DSGFT'].mean())
    new_params['ADAH'] = float32(row['ADAH'].mean())
    new_params['e_DAP'] = float32(row['e_DAP'].mean())
    new_params['h_DAP'] = float32(row['h_DAP'].mean())
    new_params['a_DAP'] = float32(row['a_DAP'].mean())
    new_params['m_DAP'] = float32(row['m_DAP'].mean())
    new_params['e_SUMDTT'] = float32(row['e_SUMDTT'].mean())
    new_params['h_SUMDTT'] = float32(row['h_SUMDTT'].mean())
    new_params['a_SUMDTT'] = float32(row['a_SUMDTT'].mean())
    new_params['m_SUMDTT'] = float32(row['m_SUMDTT'].mean())
    return new_params

# --------------------- EMERGENCE
@njit('float32[:](float32, float32, float32, int32[:], int32[:], int32[:], float32, float32, int8 )',
      cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def getGDDE(P1V, P1D, VREQ, _doy, _Tn, _Tx, maxSUMDTT, latitude, Vern=0):
    # Con el TT utilizamos la función lineal por location or genotipo y obtenemos GDDE aproximado
    obsTT = 0
    _dTT = apply_GDD(_Tn, _Tx, _doy, maxSUMDTT,  maxSUMDTT,
                     snow_depth=0, Tbase=0, Topt=26, Ttop=34, Vern=Vern, 
                     P1V=P1V, P1D=P1D, VREQ=VREQ, lat=latitude)
    obsTT += _dTT.sum()
    # Old Equations
    #if (obsTT < 190):
    #    GDDE = 0.1496304 * obsTT - 1.3641063
    #elif ((obsTT > 190) and (obsTT < 400)):
    #    GDDE = 0.0989615 * obsTT - 4.2659372
    #elif (obsTT > 400):
    #    GDDE = 0.0655266 * obsTT - 2.5076657
    GDDE = 0.2035874 * obsTT - 6.9305215 #0.2144368 * obsTT - 7.5979943 #0.2328013 * obsTT - 8.9876024
    return np.asarray([GDDE, obsTT]).astype(numba.float32)

''' Estimate emergence using a linear correlation among Thermal time and GDDE '''
def estimate_emergence_by_default_v3(sowing_date=None, latitude=None, longitude=None,  
                                     params=None, DOYs=None, Tn=None, Tx=None, config=None):
    if (sowing_date is None or latitude is None or longitude is None or config is None 
        or params is None or DOYs is None or Tn is None or Tx is None):
        return params
    # Extract values from configuration file "configbycoords" using K-Nearest neighbors
    new_params = get_calibrated_params(sowing_date=sowing_date, latitude=latitude, longitude=longitude, params=params, config=config)
    maxSUMDTT = new_params['e_SUMDTT']
    # Con el TT utilizamos la función lineal por location or genotipo y obtenemos GDDE aproximado
    arr_res = getGDDE(new_params['P1V'], new_params['P1D'], new_params['VREQ'], DOYs, Tn, Tx, maxSUMDTT, latitude, Vern=0)
    GDDE = arr_res[0] 
    obsTT = int(arr_res[1])
    new_params['GDDE'] = GDDE
    return new_params #, obsTT


# --------------------- HEADING
@njit('float32[:](float32, float32, float32, int32[:], int32[:], int32[:], float32, float32, int8 )',
      cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def getPHINT(P1V, P1D, VREQ, _doy, _Tn, _Tx, maxSUMDTT, latitude, Vern=1):
    obsTT = 0
    _dTT = apply_GDD(_Tn, _Tx, _doy, maxSUMDTT,  maxSUMDTT,
                     snow_depth=0, Tbase=0, Topt=26, Ttop=34, Vern=Vern, 
                     P1V=P1V, P1D=P1D, VREQ=VREQ, lat=latitude)
    obsTT += _dTT.sum()
    # y = 0.3020005x + 6.4609517
    PHINT = 0.3020005 * obsTT + 6.4609517 #0.3222210 * obsTT + 0.0984368
    return np.asarray([PHINT, obsTT]).astype(numba.float32)
    
''' Estimate heading using a linear correlation among Thermal time and PHINT '''
def estimate_heading_by_default_v3(sowing_date=None, latitude=None, longitude=None,  
                                     params=None, DOYs=None, Tn=None, Tx=None, config=None):
    if (sowing_date is None or latitude is None or longitude is None or config is None 
        or params is None or DOYs is None or Tn is None or Tx is None):
        return params
    # Extract values from configuration file "configbycoords" using K-Nearest neighbors
    new_params = get_calibrated_params(sowing_date=sowing_date, latitude=latitude, longitude=longitude, params=params, config=config)
    maxSUMDTT = new_params['h_SUMDTT']
    # Con el TT utilizamos la función lineal por location or genotipo y obtenemos PHINT aproximado
    arr_res = getPHINT(new_params['P1V'], new_params['P1D'], new_params['VREQ'], DOYs, Tn, Tx, maxSUMDTT, latitude, Vern=1)
    PHINT = arr_res[0] 
    obsTT = int(arr_res[1])
    new_params['PHINT'] = PHINT
    return new_params #, obsTT


# --------------------- MATURITY
@njit('float32[:](float32, float32, float32, int32[:], int32[:], int32[:], float32, float32, int8 )',
      cache=True, boundscheck=False, fastmath=True, parallel=False, nogil=True, nopython=True)
def getP5(P1V, P1D, VREQ, _doy, _Tn, _Tx, maxSUMDTT, latitude, Vern=0):
    obsTT = 0
    _dTT = apply_GDD(_Tn, _Tx, _doy, maxSUMDTT,  maxSUMDTT,
                     snow_depth=0, Tbase=0, Topt=26, Ttop=34, Vern=Vern, 
                     P1V=P1V, P1D=P1D, VREQ=VREQ, lat=latitude)
    obsTT += _dTT.sum()
    # params['P5'] = 0.9606581 * obsTT + 9.2938099
    # y = 0.9584606x + 9.8727432
    P5 = 0.9584606 * obsTT + 9.8727432 #0.9606315 * obsTT + 8.9587941
    return np.asarray([P5, obsTT]).astype(numba.float32)
    
''' Estimate maturity using a linear correlation among Thermal time and P5 '''
def estimate_maturity_by_default_v3(sowing_date=None, latitude=None, longitude=None,  
                                     params=None, DOYs=None, Tn=None, Tx=None, config=None):
    if (sowing_date is None or latitude is None or longitude is None or config is None 
        or params is None or DOYs is None or Tn is None or Tx is None):
        return params
    # Extract values from configuration file "configbycoords" using K-Nearest neighbors
    new_params = get_calibrated_params(sowing_date=sowing_date, latitude=latitude, longitude=longitude, params=params, config=config)
    maxSUMDTT = new_params['m_SUMDTT']
    # Con el TT utilizamos la función lineal por location or genotipo y obtenemos P5 aproximado
    arr_res = getP5(new_params['P1V'], new_params['P1D'], new_params['VREQ'], DOYs, Tn, Tx, maxSUMDTT, latitude, Vern=0)
    P5 = arr_res[0] 
    obsTT = int(arr_res[1])
    new_params['P5'] = P5
    return new_params #, obsTT

# Improved versions 
''' Get calibration parameters '''
def get_calibrated_params_v2(sowing_date=None, latitude=None, longitude=None, params=None, config=None):
    if (sowing_date is None or latitude is None or longitude is None or config is None 
        or params is None):
        return params
    # Extract values from configuration file "configbycoords" using K-Nearest neighbors
    sowing_date_ms = time.mktime(dt.datetime.strptime(sowing_date, "%Y-%m-%d").timetuple())*1000
    new_site = pd.DataFrame({"lat": [latitude], "lon": [longitude], "sowing_date": [sowing_date_ms] }) 
    nearest = config["nn"].kneighbors(new_site, n_neighbors=1, return_distance=False)
    row = config["configbycoords"].iloc[nearest[0]]
    # 'lat', 'lon', 'sowing_date', 'smonth', 'SDEPTH', 'GDDE', 'SNOW', 'VREQ',
    # 'PHINT', 'P1V', 'P1D', 'P5', 'P6', 'DSGFT', 'ADAH', 'e_DAP', 'h_DAP',
    # 'a_DAP', 'm_DAP', 'e_SUMDTT', 'h_SUMDTT', 'a_SUMDTT', 'm_SUMDTT'
    new_params = params.copy()
    new_params['SDEPTH'] = float32(row['SDEPTH'].mean())
    new_params['GDDE'] = float32(row['GDDE'].mean())
    new_params['SNOW'] = float32(row['SNOW'].mean())
    new_params['VREQ'] = float32(row['VREQ'].mean())
    new_params['PHINT'] = float32(row['PHINT'].mean())
    new_params['P1V'] = float32(row['P1V'].mean())
    new_params['P1D'] = float32(row['P1D'].mean())
    new_params['P5'] = float32(row['P5'].mean())
    new_params['P6'] = float32(row['P6'].mean())
    new_params['DSGFT'] = float32(row['DSGFT'].mean())
    new_params['ADAH'] = float32(row['ADAH'].mean())
    new_params['TT_EMERGENCE_LIMIT'] = float32(row['TT_EMERGENCE_LIMIT'].mean())
    new_params['TT_TBASE'] = float32(row['TT_TBASE'].mean())
    new_params['TT_TDU_LIMIT'] = float32(row['TT_TDU_LIMIT'].mean())
    new_params['e_DAP'] = float32(row['e_DAP'].mean())
    new_params['h_DAP'] = float32(row['h_DAP'].mean())
    new_params['a_DAP'] = float32(row['a_DAP'].mean())
    new_params['m_DAP'] = float32(row['m_DAP'].mean())
    new_params['e_SUMDTT'] = float32(row['e_SUMDTT'].mean())
    new_params['h_SUMDTT'] = float32(row['h_SUMDTT'].mean())
    new_params['a_SUMDTT'] = float32(row['a_SUMDTT'].mean())
    new_params['m_SUMDTT'] = float32(row['m_SUMDTT'].mean())
    return new_params

''' Estimate phenological stages using calibrated values '''
def apply_pheno_v2(df_IWIN, arr_final, config):
    params = init_params() #initparams
    # Run model for each site
    res_arr = []
    params_arr = []
    sz = 400
    idx_data = 34
    # Warm up
    for i in range(len(arr_final[:100])):
        UID = int32(arr_final[i][0])
        location = int32(arr_final[i][3]) #location
        sowing_date = arr_final[i][10] #sowing
        lat = float32(arr_final[i][17])
        lon = float32(arr_final[i][18])
        ObsEmergenceDAP = int32(arr_final[i][25])
        ObsHeadingDAP = int32(arr_final[i][26])
        ObsAnthesisDAP = int32(arr_final[i][27])
        ObsMaturityDAP = int32(arr_final[i][28])
        wdates = list(np.asarray(arr_final[i][idx_data:sz*1+idx_data], dtype='<U32'))
        DOYs = np.asarray(arr_final[i][sz*1+idx_data:sz*2+idx_data], dtype='int32')
        Tn = np.asarray(arr_final[i][sz*2+idx_data:sz*3+idx_data], dtype='int32')
        Tx = np.asarray(arr_final[i][sz*3+idx_data:sz*4+idx_data], dtype='int32')
        # Extract values from configuration file "configbycoords" using K-Nearest neighbors
        new_params = get_calibrated_params_v2(sowing_date=sowing_date, latitude=lat, longitude=lon, 
                                           params=params, config=config)
        R = numba.typed.List.empty_list(numba.types.string)
        _ = run_pheno_model(i, new_params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, R)
    # Run models
    for i in tqdm(range(len(arr_final))):
        UID = int32(arr_final[i][0])
        location = int32(arr_final[i][3]) #location
        sowing_date = arr_final[i][10] #sowing
        lat = float32(arr_final[i][17])
        lon = float32(arr_final[i][18])
        ObsEmergenceDAP = int32(arr_final[i][25])
        ObsHeadingDAP = int32(arr_final[i][26])
        ObsAnthesisDAP = int32(arr_final[i][27])
        ObsMaturityDAP = int32(arr_final[i][28])
        
        wdates = list(np.asarray(arr_final[i][idx_data:sz*1+idx_data], dtype='<U32'))
        DOYs = np.asarray(arr_final[i][sz*1+idx_data:sz*2+idx_data], dtype='int32')
        Tn = np.asarray(arr_final[i][sz*2+idx_data:sz*3+idx_data], dtype='int32')
        Tx = np.asarray(arr_final[i][sz*3+idx_data:sz*4+idx_data], dtype='int32')
        #rhx = np.asarray(arr_final[i][sz*4+idx_data:sz*5+idx_data], dtype='int32')
        #pcp = np.asarray(arr_final[i][sz*5+idx_data:sz*6+idx_data], dtype='int32')
        #srad = np.asarray(arr_final[i][sz*6+idx_data:sz*7+idx_data], dtype='int32')
        #

        # Get Calibrated Parameters
        # Extract values from configuration file "configbycoords" using K-Nearest neighbors
        new_params = get_calibrated_params_v2(sowing_date=sowing_date, latitude=lat, longitude=lon, 
                                           params=params, config=config)
        
        #e_params = estimate_emergence_by_default_v3(sowing_date=sowing_date, latitude=lat, longitude=lon,  
        #                             params=params, DOYs=DOYs, Tn=Tn, Tx=Tx, config=config)
        #h_params = estimate_heading_by_default_v3(sowing_date=sowing_date, latitude=lat, longitude=lon,  
        #                             params=e_params, DOYs=DOYs, Tn=Tn, Tx=Tx, config=config)
        #m_params = estimate_heading_by_default_v3(sowing_date=sowing_date, latitude=lat, longitude=lon,  
        #                             params=h_params, DOYs=DOYs, Tn=Tn, Tx=Tx, config=config)
        
        RESULTS = numba.typed.List.empty_list(numba.types.string)
        _arr = run_pheno_model(UID, new_params, wdates, location, sowing_date, 
                                                ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                                lat, lon, DOYs, Tn, Tx, RESULTS)
        # save params
        params_arr.append({'UID':UID, 'params':new_params})
        res_arr.append([str(x) for x in _arr]) # Covert from numba.List to numpy array and append
    
    #
    if (len(res_arr)>0):
        ph = pd.DataFrame(res_arr)
        ph.columns = ['UID','location','Sowing', 'SimGermination', 'g_DOY', 'g_DAP', 'g_AGE', 'g_SUMDTT',
                      'SimEmergence', 'e_DOY', 'e_DAP', 'e_AGE', 'e_SUMDTT',
                      'Sim Term Spklt', 'ts_DOY', 'ts_DAP', 'ts_AGE', 'ts_SUMDTT',
                      'Sim End Veg', 'ev_DOY', 'ev_DAP', 'ev_AGE', 'ev_SUMDTT',
                      'SimHeading', 'h_DOY', 'h_DAP', 'h_AGE', 'h_SUMDTT', #'End Ear Gr or Heading'
                      'SimAnthesis', 'a_DOY', 'a_DAP', 'a_AGE', 'a_SUMDTT', 
                      'Sim End Pan Gr', 'epg_DOY', 'epg_DAP', 'epg_AGE', 'epg_SUMDTT',
                      'Sim Beg Gr Fil', 'bgf_DOY', 'bgf_DAP', 'bgf_AGE', 'bgf_SUMDTT',
                      'SimMaturity', 'm_DOY', 'm_DAP', 'm_AGE', 'm_SUMDTT', #'End Gr Fil', 
                      'SimHarvest', 'hv_DOY', 'hv_DAP', 'hv_AGE', 'hv_SUMDTT', 
                      #'e_status', 'h_status', 'a_status', 'm_status', 
                      #'TT_EMERGENCE_LIMIT', 'TT_TDU_LIMIT',
                      #'SDEPTH', 'GDDE', 'SNOW', 'VREQ', 'PHINT', 'P1V', 'P1D', 'DSGFT', 'P5', 'P6', 'ADAH'
                     ]
        # Try to convert formats
        try:
            dates_columns = ['Sowing', 'SimGermination', 'SimEmergence', 'Sim Term Spklt', 'Sim End Veg', 'SimHeading', #'End Ear Gr or Heading'
                              'SimAnthesis', 'Sim End Pan Gr','Sim Beg Gr Fil', 'SimMaturity', #'End Gr Fil', 
                              'SimHarvest']
            
            ph[dates_columns] = ph[dates_columns].apply(lambda x: pd.to_datetime(x.astype(str), format="%Y-%m-%d"), axis=0 )
            
        except Exception as err:
            pass
        try:
            intcols = ['UID','location']
            ph[intcols] = ph[intcols].astype(np.uint32)
            fcols = ['g_DOY', 'g_DAP', 'g_AGE', 'g_SUMDTT',
                      'e_DOY', 'e_DAP', 'e_AGE', 'e_SUMDTT',
                      'ts_DOY', 'ts_DAP', 'ts_AGE', 'ts_SUMDTT',
                      'ev_DOY', 'ev_DAP', 'ev_AGE', 'ev_SUMDTT',
                      'h_DOY', 'h_DAP', 'h_AGE', 'h_SUMDTT', #'End Ear Gr or Heading'
                      'a_DOY', 'a_DAP', 'a_AGE', 'a_SUMDTT', 
                      'epg_DOY', 'epg_DAP', 'epg_AGE', 'epg_SUMDTT',
                      'bgf_DOY', 'bgf_DAP', 'bgf_AGE', 'bgf_SUMDTT',
                      'm_DOY', 'm_DAP', 'm_AGE', 'm_SUMDTT', #'End Gr Fil', 
                      'hv_DOY', 'hv_DAP', 'hv_AGE', 'hv_SUMDTT', 
                      #'e_status', 'h_status', 'a_status', 'm_status',
                      #'TT_EMERGENCE_LIMIT', 'TT_TDU_LIMIT',
                      #'SDEPTH', 'GDDE', 'SNOW', 'VREQ', 'PHINT', 'P1V', 'P1D', 'DSGFT', 'P5', 'P6', 'ADAH'
                    ]
            ph[fcols] = ph[fcols].astype(np.float32)
        except Exception as err:
            pass
        
        try:
            df = pd.merge(df_IWIN[['UID','location', 'lat', 'lon', 'sowing', 'emergence', 'heading', 'maturity',
                          'ObsEmergDOY','ObsEmergDAP', 'ObsHeadingDOY', 'ObsHeadingDAP', 'ObsAnthesisDOY', 'ObsAnthesisDAP',
                          'ObsMaturityDAP', 'ObsMaturityDOY'
                         ]], ph, on=['UID'], how='right',
                )
        
            df['eresidualsDAP'] = df['ObsEmergDAP'] - df['e_DAP']
            df['eresidualsDOY'] = df['ObsEmergDOY'] - df['e_DOY']
            df['hresidualsDAP'] = df['ObsHeadingDAP'] - df['h_DAP']
            df['hresidualsDOY'] = df['ObsHeadingDOY'] - df['h_DOY']
            df['aresidualsDAP'] = df['ObsAnthesisDAP'] - df['a_DAP']
            df['aresidualsDOY'] = df['ObsAnthesisDOY'] - df['a_DOY']
            df['mresidualsDAP'] = df['ObsMaturityDAP'] - df['m_DAP']
            df['mresidualsDOY'] = df['ObsMaturityDOY'] - df['m_DOY']
            df.rename(columns={'location_x': 'location'}, inplace=True)
            df.drop(columns=['location_y', 'Sowing'], inplace=True)
            hoy = dt.datetime.now().strftime('%Y%m%d')
            df.to_parquet(f'df_pywheat_iwin_phenology_{hoy}.parquet')
            return df
        except Exception as err:
            pass
    return ph
    
# -------------------------------------------
# -------------------------------------------
# Plot Calibration
# -------------------------------------------
def plot_results_v3(df_tmp=None, xy_lim=40, xylim=False, clr1='b',clr2='r', clr3='black',s=5,alpha=0.2,
                     e_threshold=10, h_threshold=15, a_threshold=10, m_threshold=10,
                     rmoutliers=False, dispScore=True, dispBads=False, fgsize=(8,14), 
                     title='Phenological stages of Wheat (IWIN)',
                     saveFig=True, showFig=True, path_to_save_results='./', dirname='Figures', 
                     fname='Fig_1_calibration_phenology_IWIN_locations', fmt='pdf' ):

    def createFigure_v2(ax, data=None, code=(-1,1), stage='', v='DOY', fld1=None, fld2=None, 
                     xy_lim=40, clr1='g', clr2='r', clr3='black', s=10, alpha=0.2, 
                     dispScore=True, dispBads=False, xylim=True):
        df = data.copy()
        df.dropna(subset=[fld1,fld2], inplace=True)
        df_ok = df[df['status']==code[1]]
        df_outliers = df[df['status']==code[0]]
        df_outliers2 = df[( (df['status']==-99) | (df['status']==0) )]
        ax.scatter(x=df_ok[fld1].to_numpy(), y=df_ok[fld2].to_numpy(), label=f'{stage} {v} (Wheat)', color=clr1, s=s, alpha=alpha )
        ax.scatter(x=df_outliers[fld1].to_numpy(), y=df_outliers[fld2].to_numpy(), label='Outliers', color=clr2, s=s, alpha=alpha, zorder=-1 )
        if (dispBads is True):
            ax.scatter(x=df_outliers2[fld1].to_numpy(), y=df_outliers2[fld2].to_numpy(), label='Outliers 2', color=clr3, s=s, alpha=alpha )
        ax.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=-2, label="line 1:1") #c=".5",
        #if (xylim is True):
        #    #maxlim = int(max(df[fld1].max(), df[fld2].max())) + xy_lim
        #    #ax.set(xlim=(0, maxlim), ylim=(0, maxlim))
        #    pass
        ax.set_xlabel(f"Observed {stage} ({v})")
        ax.set_ylabel(f"Simulated {stage} ({v})")
        ax.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.25)
        ax.set_axisbelow(True)
        if (dispScore==True):
            try:
                r2score0, mape0, rmse0, n_rmse0, d_index0, ef0, accuracy0 = getScores(df, fld1=fld1, fld2=fld2)
                if (rmoutliers is True):
                    r2score, mape, rmse, n_rmse, d_index, ef, accuracy = getScores(df_ok, fld1=fld1, fld2=fld2)
                    ax.text(0.05,0.96,'Observations: {}\nRMSE: {:.1f} - [{:.1f}] days'.format(len(df), rmse0, rmse) + '\nNRMSE: {:.3f} - [{:.3f}]\nd-index: {:.2f} - [{:.2f}]\nR$^2$: {:.2f} - [{:.2f}]\nAccuracy: {:.2f}% - [{:.2f}%]'.format(n_rmse0, n_rmse, d_index0, d_index, 
                                                                                                                                                                                                                                           r2score0, r2score, accuracy0, accuracy), 
                             fontsize=8, ha='left', va='top', transform=ax.transAxes)
                else:
                    r2score, mape, rmse, n_rmse, d_index, ef, accuracy = getScores(df_ok, fld1=fld1, fld2=fld2)
                    ax.text(0.05,0.96,'Observations: {}\nOutliers: {}\nRMSE: {:.1f} - [{:.1f}] days'.format(len(df), (len(df_outliers) + len(df_outliers2)), rmse0, rmse) + '\nNRMSE: {:.3f} - [{:.3f}]\nd-index: {:.2f} - [{:.2f}]\nR$^2$: {:.2f} - [{:.2f}]\nAccuracy: {:.2f}% - [{:.2f}%]'.format(n_rmse0, n_rmse, d_index0, d_index, 
                                                                                                                                                                                                                                           r2score0, r2score, accuracy0, accuracy), 
                             fontsize=8, ha='left', va='top', transform=ax.transAxes)
            except Exception as err:
                #print("Problem getting metrics")
                pass

    # Create figures
    df = df_tmp.copy()
    df['status']=0
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2, figsize=fgsize)
    fig.subplots_adjust(top=0.9)

    # Mejorar la visualización de outliers
    df_final = pd.DataFrame()
    # Residuals
    if (('ObsEmergDOY' in df.columns) or ('ObsEmergDAP' in df.columns) ):
        # Figures 1
        df_final = pd.concat([df_final, df[df['e_status']==1]], axis=0)
        df_e = df[~df['ObsEmergDAP'].isnull()]
        df_e['status']= df_e['e_status']
        if (rmoutliers is True):
            df_e = df_e[~( (df_e['ObsEmergDAP']>40) | (df_e['e_DAP']>40) )]
            df_e = df_e[~( (df_e['status']!=1) & (df_e['eresidualsDAP']< -e_threshold) & (df_e['eresidualsDAP'] > e_threshold) )]
        else:
            df_e.loc[ ( (df_e['ObsEmergDAP']>40) | (df_e['e_DAP']>40) ), 'status'] = -99
            df_e.loc[( (df_e['status']!=1) & (df_e['eresidualsDAP']> -e_threshold) & (df_e['eresidualsDAP'] < e_threshold) ), 'status'] = -1
        
        createFigure_v2(ax1, data=df_e, code=(-1,1), stage='Emergence', v='DOY', fld1="ObsEmergDOY", fld2="e_DOY", 
                     xy_lim=40, clr1='brown', clr2='b', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figures 2
        createFigure_v2(ax2, data=df_e, code=(-1,1), stage='Emergence', v='DAP', fld1="ObsEmergDAP", fld2="e_DAP", 
                     xy_lim=40, clr1='brown', clr2='b', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    if (('ObsHeadingDOY' in df.columns) or ('ObsHeadingDAP' in df.columns)):
        # Figures 3 
        df_final = pd.concat([df_final, df[df['h_status']==2]], axis=0)
        df_h = df[~df['ObsHeadingDAP'].isnull()]
        df_h['status']= df_h['h_status']
        if (rmoutliers is True):
            df_h = df_h[~( (df_h['status']!=2) & (df_h['hresidualsDAP'] < -h_threshold) & (df_h['hresidualsDAP'] > h_threshold) )]
        else:
            df_h.loc[( (df_h['status']!=2) & (df_h['hresidualsDAP']> -h_threshold) & (df_h['hresidualsDAP'] < h_threshold) ), 'status'] = -2
        
        createFigure_v2(ax3, data=df_h, code=(-2,2), stage='Heading', v='DOY', fld1="ObsHeadingDOY", fld2="h_DOY", 
                     xy_lim=40, clr1='g', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figures 4
        createFigure_v2(ax4, data=df_h, code=(-2,2), stage='Heading', v='DAP', fld1="ObsHeadingDAP", fld2="h_DAP", 
                     xy_lim=40, clr1='g', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    if (('ObsAnthesisDOY' in df.columns) or ('ObsAnthesisDAP' in df.columns)):
        # Figures 5
        df_final = pd.concat([df_final, df[df['a_status']==3]], axis=0)
        df_a = df[~df['ObsAnthesisDAP'].isnull()]
        df_a['status']= df_a['a_status']
        if (rmoutliers is True):
            df_a = df_a[~( (df_a['status']!=3) & (df_a['aresidualsDAP'] < -a_threshold) & (df_a['aresidualsDAP'] > a_threshold) )]
        else:
            df_a.loc[( (df_a['status']!=3) & (df_a['aresidualsDAP']> -a_threshold) & (df_a['aresidualsDAP'] < a_threshold) ), 'status'] = -3
        
        createFigure_v2(ax5, data=df_a, code=(-3,3), stage='Anthesis', v='DOY', fld1="ObsAnthesisDOY", fld2="a_DOY", 
                     xy_lim=40, clr1='purple', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figure 6
        createFigure_v2(ax6, data=df_a, code=(-3,3), stage='Anthesis', v='DAP', fld1="ObsAnthesisDAP", fld2="a_DAP", 
                     xy_lim=40, clr1='purple', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    if (('ObsMaturityDOY' in df.columns) or ('ObsMaturityDAP' in df.columns)):
        # Figures 7
        df_final = pd.concat([df_final, df[df['m_status']==4]], axis=0)
        df_m = df[~df['ObsMaturityDAP'].isnull()]
        df_m['status']= df_m['m_status']
        if (rmoutliers is True):
            df_m = df_m[~( (df_m['status']!=4) & (df_m['mresidualsDAP'] < -m_threshold) & (df_m['mresidualsDAP'] > m_threshold) )]
        else:
            df_m.loc[( (df_m['status']!=4) & (df_m['mresidualsDAP']> -m_threshold) & (df_m['mresidualsDAP'] < m_threshold) ), 'status'] = -4
        
        createFigure_v2(ax7, data=df_m, code=(-4,4), stage='Maturity', v='DOY', fld1="ObsMaturityDOY", fld2="m_DOY", 
                     xy_lim=40, clr1='orange', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figures 8
        createFigure_v2(ax8, data=df_m, code=(-4,4), stage='Maturity', v='DAP', fld1="ObsMaturityDAP", fld2="m_DAP", 
                     xy_lim=40, clr1='orange', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    fig.suptitle(f'{title}', fontsize=18) #
    fig.tight_layout()
    # Save in PDF
    hoy = dt.datetime.now().strftime('%Y%m%d')
    figures_path = os.path.join(path_to_save_results, '{}_{}'.format(dirname, hoy))
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    if (saveFig is True and fmt=='pdf'):
        fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                    bbox_inches='tight', orientation='portrait',  
                    edgecolor='none', transparent=False, pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)

    #if (showFig is True):
    #    fig.show()
    #else:
    #    del fig
    #    plt.close();
    return fig, df_final


def plot_pheno_accuracy(df_tmp=None, xy_lim=40, xylim=False, clr1='b',clr2='r', clr3='black',s=5,alpha=0.2,
                     e_threshold=10, h_threshold=15, a_threshold=10, m_threshold=10,
                     rmoutliers=False, dispScore=True, dispBads=False, fgsize=(8,14), 
                     title='Phenological stages of Wheat (IWIN)',
                     saveFig=True, showFig=True, path_to_save_results='./', dirname='Figures', 
                     fname='Fig_1_phenology_model_result', fmt='jpg' ):

    def createFigure_v3(ax, data=None, code=(-1,1), stage='', v='DOY', fld1=None, fld2=None, 
                     xy_lim=40, clr1='g', clr2='r', clr3='black', s=10, alpha=0.2, 
                     dispScore=True, dispBads=False, xylim=True):
        df = data.copy()
        df.dropna(subset=[fld1,fld2], inplace=True)
        df_ok = df[df['status']==code[1]]
        df_outliers = df[df['status']==code[0]]
        df_outliers2 = df[( (df['status']==-99) | (df['status']==0) )]
        ax.scatter(x=df_ok[fld1].to_numpy(), y=df_ok[fld2].to_numpy(), label=f'{stage} {v} (Wheat)', color=clr1, s=s, alpha=alpha )
        ax.scatter(x=df_outliers[fld1].to_numpy(), y=df_outliers[fld2].to_numpy(), label='Outliers', color=clr2, s=s, alpha=alpha, zorder=-1 )
        if (dispBads is True):
            ax.scatter(x=df_outliers2[fld1].to_numpy(), y=df_outliers2[fld2].to_numpy(), label='Outliers 2', color=clr3, s=s, alpha=alpha )
        ax.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=-2, label="line 1:1") #c=".5",
        #if (xylim is True):
        #    #maxlim = int(max(df[fld1].max(), df[fld2].max())) + xy_lim
        #    #ax.set(xlim=(0, maxlim), ylim=(0, maxlim))
        #    pass
        ax.set_xlabel(f"Observed {stage} ({v})")
        ax.set_ylabel(f"Simulated {stage} ({v})")
        ax.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.25)
        ax.set_axisbelow(True)
        if (dispScore==True):
            try:
                r2score0, mape0, rmse0, n_rmse0, d_index0, ef0, accuracy0 = getScores(df, fld1=fld1, fld2=fld2)
                if (rmoutliers is True):
                    r2score, mape, rmse, n_rmse, d_index, ef, accuracy = getScores(df_ok, fld1=fld1, fld2=fld2)
                    ax.text(0.05,0.96,'Observations: {}\nRMSE: {:.1f} days'.format(len(df), rmse) + '\nNRMSE: {:.3f}\nd-index: {:.2f}\nR$^2$: {:.2f}\nAccuracy: {:.2f}%'.format(n_rmse, d_index, r2score, accuracy), 
                             fontsize=8, ha='left', va='top', transform=ax.transAxes)
                else:
                    r2score, mape, rmse, n_rmse, d_index, ef, accuracy = getScores(df_ok, fld1=fld1, fld2=fld2)
                    ax.text(0.05,0.96,'Observations: {}\nOutliers: {}\nRMSE: {:.1f} - [{:.1f}] days'.format(len(df), (len(df_outliers)), rmse0, rmse) + '\nNRMSE: {:.3f} - [{:.3f}]\nd-index: {:.2f} - [{:.2f}]\nR$^2$: {:.2f} - [{:.2f}]\nAccuracy: {:.2f}% - [{:.2f}%]'.format(n_rmse0, n_rmse, d_index0, d_index, 
                                                                                                                                                                                                                                           r2score0, r2score, accuracy0, accuracy), 
                             fontsize=8, ha='left', va='top', transform=ax.transAxes)
            except Exception as err:
                #print("Problem getting metrics")
                pass

    # Create figures
    df = df_tmp.copy()
    df['status']=0
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2, figsize=fgsize)
    fig.subplots_adjust(top=0.9)

    # Mejorar la visualización de outliers
    # Residuals
    if (('ObsEmergDOY' in df.columns) or ('ObsEmergDAP' in df.columns) ):
        # Figures 1
        df_e = df[~df['ObsEmergDAP'].isnull()]
        df_e['status']= 1
        if (rmoutliers is True):
            df_e = df_e[~( (df_e['ObsEmergDOY']>200) & (df_e['e_DOY']<100) )]
            df_e = df_e[~( (df_e['ObsEmergDOY']<100) & (df_e['e_DOY']>200) )]
            df_e = df_e[~( (df_e['ObsEmergDAP']>40) | (df_e['e_DAP']>40) )]
            df_e = df_e[~( (df_e['eresidualsDAP']< -e_threshold) | (df_e['eresidualsDAP'] > e_threshold) )]
        else:
            df_e.loc[( (df_e['eresidualsDAP']< -e_threshold) | (df_e['eresidualsDAP'] > e_threshold) ), 'status'] = -1
            df_e.loc[( (df_e['ObsEmergDAP']>40) | (df_e['e_DAP']>40) ), 'status'] = -99
            df_e.loc[( (df_e['ObsEmergDOY']>200) & (df_e['e_DOY']<100) ), 'status'] = -99
            df_e.loc[( (df_e['ObsEmergDOY']<100) & (df_e['e_DOY']>200) ), 'status'] = -99
        
        createFigure_v3(ax1, data=df_e, code=(-1,1), stage='Emergence', v='DOY', fld1="ObsEmergDOY", fld2="e_DOY", 
                     xy_lim=40, clr1='brown', clr2='b', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figures 2
        createFigure_v3(ax2, data=df_e, code=(-1,1), stage='Emergence', v='DAP', fld1="ObsEmergDAP", fld2="e_DAP", 
                     xy_lim=40, clr1='brown', clr2='b', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    if (('ObsHeadingDOY' in df.columns) or ('ObsHeadingDAP' in df.columns)):
        # Figures 3 
        df_h = df[~df['ObsHeadingDAP'].isnull()]
        df_h['status']= 2
        if (rmoutliers is True):
            df_h = df_h[~( (df_h['ObsHeadingDOY']>200) & (df_h['h_DOY']<100) )]
            df_h = df_h[~( (df_h['ObsHeadingDOY']<100) & (df_h['h_DOY']>200) )]
            df_h = df_h[~( (df_h['hresidualsDAP'] < -h_threshold) | (df_h['hresidualsDAP'] > h_threshold) )]
        else:
            df_h.loc[( (df_h['hresidualsDAP'] < -h_threshold) | (df_h['hresidualsDAP'] > h_threshold) ), 'status'] = -2
            df_h.loc[( (df_h['ObsHeadingDOY']>200) & (df_h['h_DOY']<100) ), 'status'] = -99
            df_h.loc[( (df_h['ObsHeadingDOY']<100) & (df_h['h_DOY']>200) ), 'status'] = -99
        createFigure_v3(ax3, data=df_h, code=(-2,2), stage='Heading', v='DOY', fld1="ObsHeadingDOY", fld2="h_DOY", 
                     xy_lim=40, clr1='g', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figures 4
        createFigure_v3(ax4, data=df_h, code=(-2,2), stage='Heading', v='DAP', fld1="ObsHeadingDAP", fld2="h_DAP", 
                     xy_lim=40, clr1='g', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    if (('ObsAnthesisDOY' in df.columns) or ('ObsAnthesisDAP' in df.columns)):
        # Figures 5
        df_a = df[~df['ObsAnthesisDAP'].isnull()]
        df_a['status']= 3
        if (rmoutliers is True):
            df_a = df_a[~( (df_a['ObsAnthesisDOY']>200) & (df_a['a_DOY']<100) )]
            df_a = df_a[~( (df_a['ObsAnthesisDOY']<100) & (df_a['a_DOY']>200) )]
            df_a = df_a[~( (df_a['aresidualsDAP'] < -a_threshold) | (df_a['aresidualsDAP'] > a_threshold) )]
        else:
            df_a.loc[( (df_a['aresidualsDAP'] < -a_threshold) | (df_a['aresidualsDAP'] > a_threshold) ), 'status'] = -3
            df_a.loc[( (df_a['ObsAnthesisDOY']>200) & (df_a['a_DOY']<100) ), 'status'] = -99
            df_a.loc[( (df_a['ObsAnthesisDOY']<100) & (df_a['a_DOY']>200) ), 'status'] = -99
            
        createFigure_v3(ax5, data=df_a, code=(-3,3), stage='Anthesis', v='DOY', fld1="ObsAnthesisDOY", fld2="a_DOY", 
                     xy_lim=40, clr1='purple', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figure 6
        createFigure_v3(ax6, data=df_a, code=(-3,3), stage='Anthesis', v='DAP', fld1="ObsAnthesisDAP", fld2="a_DAP", 
                     xy_lim=40, clr1='purple', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    if (('ObsMaturityDOY' in df.columns) or ('ObsMaturityDAP' in df.columns)):
        # Figures 7
        df_m = df[~df['ObsMaturityDAP'].isnull()]
        df_m['status']= 4
        if (rmoutliers is True):
            df_m = df_m[~( (df_m['ObsMaturityDOY']>200) & (df_a['m_DOY']<100) )]
            df_m = df_m[~( (df_m['ObsMaturityDOY']<100) & (df_a['m_DOY']>200) )]
            df_m = df_m[~( (df_m['mresidualsDAP']< -m_threshold) | (df_m['mresidualsDAP'] > m_threshold) )]
        else:
            df_m.loc[( (df_m['mresidualsDAP']< -m_threshold) | (df_m['mresidualsDAP'] > m_threshold) ), 'status'] = -4
            df_m.loc[( (df_m['ObsMaturityDOY']>200) & (df_m['m_DOY']<100) ), 'status'] = -99
            df_m.loc[( (df_m['ObsMaturityDOY']<100) & (df_m['m_DOY']>200) ), 'status'] = -99
        
        createFigure_v3(ax7, data=df_m, code=(-4,4), stage='Maturity', v='DOY', fld1="ObsMaturityDOY", fld2="m_DOY", 
                     xy_lim=40, clr1='orange', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)
        # Figures 8
        createFigure_v3(ax8, data=df_m, code=(-4,4), stage='Maturity', v='DAP', fld1="ObsMaturityDAP", fld2="m_DAP", 
                     xy_lim=40, clr1='orange', s=s, alpha=alpha, dispScore=dispScore, dispBads=dispBads, xylim=True)

    fig.suptitle(f'{title}', fontsize=18) #
    fig.tight_layout()
    # Save in PDF
    hoy = dt.datetime.now().strftime('%Y%m%d')
    figures_path = os.path.join(path_to_save_results, '{}_{}'.format(dirname, hoy))
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    if (saveFig is True and fmt=='pdf'):
        fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                    bbox_inches='tight', orientation='portrait',  
                    edgecolor='none', transparent=False, pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)

    #if (showFig is True):
    #    fig.show()
    #else:
    #    del fig
    #    plt.close();
    return fig

# -------------------------------------------
#
def createCalibration_Curves(dfcal):
    sel_cols = ['lat', 'lon', 'sowing', 'SDEPTH', 'GDDE', 'SNOW', 'VREQ',
                'PHINT', 'P1V', 'P1D', 'P5', 'P6', 'DSGFT', 'ADAH', 'TT_EMERGENCE_LIMIT', 'TT_TBASE', 'TT_TDU_LIMIT',
                'e_DAP', 'h_DAP', 'a_DAP', 'm_DAP', 'e_SUMDTT', 'h_SUMDTT', 'a_SUMDTT', 'm_SUMDTT',
                'e_status', 'h_status', 'a_status', 'm_status'
               ]
    config_parameters_byLatLng = dfcal[sel_cols].copy()
    config_parameters_byLatLng = config_parameters_byLatLng[( (config_parameters_byLatLng['e_status']==1) 
                                                             | (config_parameters_byLatLng['h_status']==2) 
                                                             | (config_parameters_byLatLng['a_status']==3) 
                                                             | (config_parameters_byLatLng['m_status']==4) 
                                                   )]
    config_parameters_byLatLng['sowing_date'] = config_parameters_byLatLng['sowing'].apply(lambda x: time.mktime(x.timetuple())*1000 )
    config_parameters_byLatLng['smonth'] = config_parameters_byLatLng['sowing'].dt.month
    config_parameters_byLatLng = config_parameters_byLatLng[['lat', 'lon', 'sowing_date', 'smonth', 'SDEPTH', 'GDDE', 'SNOW', 'VREQ',
           'PHINT', 'P1V', 'P1D', 'P5', 'P6', 'DSGFT', 'ADAH', 'TT_EMERGENCE_LIMIT', 'TT_TBASE', 'TT_TDU_LIMIT', 'e_DAP',
           'h_DAP', 'a_DAP', 'm_DAP', 'e_SUMDTT', 'h_SUMDTT', 'a_SUMDTT', 'm_SUMDTT']]
    config_parameters_byLatLng = config_parameters_byLatLng.sort_values(['lat','lon','sowing_date']).reset_index(drop=True)
    print(f"Number of Ecotypes: {len(config_parameters_byLatLng)}")
    # drop duplicates
    config_parameters_byLatLng.drop_duplicates(inplace=True)
    print(f"Number of Ecotypes (after dropped dup): {len(config_parameters_byLatLng)}")
    # Save results
    config_parameters_byLatLng.to_parquet("configbycoords_v0.0.9.cfg", index=False)
    #config_parameters_byLatLng.head()
    #
    # Charting 
    def addLinearReg(df, ax, fld1, fld2):
        # Add linear regression for GY
        df_cleaned = df.dropna(subset=[fld1, fld2]).reset_index(drop=True)
        x = df_cleaned[fld1].astype(float).to_numpy()
        y = df_cleaned[fld2].astype(float).to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        print("y = {:.7f}x + {:.7f}".format(pend, intercept))
        y_predicted = [pend*i + intercept  for i in x]
        l1 = sns.lineplot(x=x,y=y_predicted, color='blue', ax=ax, ls='--', lw=0.85) #, label='Linear Regression')
        ax.text(0.95, 0.1,r'$y$ = {:.3f}$X$ + {:.3f}'.format(pend, intercept)+'\n', #"n: " + r"$\bf" + str(len(df)) + "$" +
                #+'\nRMSE:{:.1f}'.format(rmse)+' tha$^{-1}$' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}\nAccuracy: {:.1f}%'.format(n_rmse, d_index, r2score, accuracy), 
                     fontsize=12.5, ha='right', va='top', transform=ax.transAxes)
    
    df_gen = config_parameters_byLatLng.copy()
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize=(10,10), facecolor='white')
    
    # Emergence
    fld1 = 'e_SUMDTT'
    fld2 = 'GDDE'
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df_gen, alpha=0.25, s=5, color="#444444",  ax=ax1);
    addLinearReg(df_gen, ax1, fld1, fld2)
    
    # Heading
    fld1 = 'h_SUMDTT'
    fld2 = 'PHINT'
    g2 = sns.scatterplot(x=fld1, y=fld2, data=df_gen, alpha=0.25, s=5, color="#444444",  ax=ax2);
    addLinearReg(df_gen, ax2, fld1, fld2)
    
    # Anthesis
    fld1 = 'a_SUMDTT'
    fld2 = 'ADAH' #'ADAH'
    g3 = sns.scatterplot(x=fld1, y=fld2, data=df_gen, alpha=0.25, s=5, color="#444444", ax=ax3);
    addLinearReg(df_gen, ax3, fld1, fld2)
    
    # Maturity
    fld1 = 'm_SUMDTT'
    fld2 = 'P5'
    g4 = sns.scatterplot(x=fld1, y=fld2, data=df_gen, alpha=0.25, s=5, color="#444444",  ax=ax4);
    addLinearReg(df_gen, ax4, fld1, fld2)
        
    plt.show()



# -------------------------------------------
# CLI - main.py in the root of the library
# -------------------------------------------
def run_pheno_cli(initparams, sowing_date, lat, lon, weather, config=None, best=True, 
                  fmt='txt', output=None):
    UID = int32(0)
    location = int32(1) #12302
    sowing_date = str(sowing_date)
    s_DOY = str(dt.datetime.strptime(sowing_date, "%Y-%m-%d").strftime("%j"))
    lat = float32(lat)
    lon = float32(lon)
    ObsEmergenceDAP = int32(0)
    ObsHeadingDAP = int32(0)
    ObsAnthesisDAP = int32(0)
    ObsMaturityDAP = int32(0)
    # Setup weather paramaters
    weather['DATE'] = pd.to_datetime(weather['DATE'].astype(str), format="%Y-%m-%d")
    weather['DOY'] = weather['DATE'].dt.dayofyear
    # Select from sowing date 
    e_date = (dt.datetime.strptime(sowing_date, "%Y-%m-%d") + dt.timedelta(days = 400)).strftime("%Y-%m-%d")
    w = weather[( (weather['DATE']>=sowing_date) & (weather['DATE']<e_date) )]
    wdates = list(np.asarray([ d.strftime("%Y-%m-%d") for d in w['DATE']]).astype('<U32'))
    DOYs = np.asarray(w['DOY'].to_numpy(), dtype='int32')
    Tn = np.asarray(w['TMIN'].astype(np.float32)*100).astype(np.int32)
    Tx = np.asarray(w['TMAX'].astype(np.float32)*100).astype(np.int32)
    #initparams
    params = init_params(initparams)
    if (best is True):
        if (config is not None):
            # Extract values from configuration file "configbycoords" using K-Nearest neighbors
            params = get_calibrated_params_v2(sowing_date=sowing_date, latitude=lat, longitude=lon, 
                                                        params=params, config=config)
    R = numba.typed.List.empty_list(numba.types.string)
    _arr = run_pheno_model(UID, params, wdates, location, sowing_date, 
                                     ObsEmergenceDAP, ObsHeadingDAP, ObsAnthesisDAP, ObsMaturityDAP,
                                     lat, lon, DOYs, Tn, Tx, R)
    res_arr = [str(x) for x in _arr]
    ph = None
    try:
        ph = pd.DataFrame([res_arr])
        ph.columns = ['UID','location','Sowing', 'SimGermination', 'g_DOY', 'g_DAP', 'g_AGE', 'g_SUMDTT',
                      'SimEmergence', 'e_DOY', 'e_DAP', 'e_AGE', 'e_SUMDTT',
                      'Sim Term Spklt', 'ts_DOY', 'ts_DAP', 'ts_AGE', 'ts_SUMDTT', # 'End of Juvenile',
                      'Sim End Veg', 'ev_DOY', 'ev_DAP', 'ev_AGE', 'ev_SUMDTT',
                      'SimHeading', 'h_DOY', 'h_DAP', 'h_AGE', 'h_SUMDTT', #'End Ear Gr or Heading'
                      'SimAnthesis', 'a_DOY', 'a_DAP', 'a_AGE', 'a_SUMDTT', 
                      'Sim End Pan Gr', 'epg_DOY', 'epg_DAP', 'epg_AGE', 'epg_SUMDTT',
                      'Sim Beg Gr Fil', 'bgf_DOY', 'bgf_DAP', 'bgf_AGE', 'bgf_SUMDTT',
                      'SimMaturity', 'm_DOY', 'm_DAP', 'm_AGE', 'm_SUMDTT', #'End Gr Fil', 
                      'SimHarvest', 'hv_DOY', 'hv_DAP', 'hv_AGE', 'hv_SUMDTT', 
                     ]
        if (ph is not None):
            if (fmt=='txt'):
                growstages = {
                        '7': {'istage_old': 'Sowing', 'istage': 'Fallow', 'desc': 'No crop present to Sowing', 'date':ph.iloc[0]['Sowing'], 'DOY':str(s_DOY), 'AGE':'0', 'DAP':'0', 'SUMDTT':'0'},
                        '8': {'istage_old': 'Germinate', 'istage': 'Sowing', 'desc': 'Sowing to Germination', 'date':ph.iloc[0]['SimGermination'], 'DOY':ph.iloc[0]['g_DOY'], 'AGE':ph.iloc[0]['g_AGE'], 'DAP':ph.iloc[0]['g_DAP'], 'SUMDTT':ph.iloc[0]['g_SUMDTT']},
                        '9': {'istage_old': 'Emergence', 'istage': 'Germinate', 'desc': 'Emergence to End of Juvenile', 'date':ph.iloc[0]['SimEmergence'], 'DOY':ph.iloc[0]['e_DOY'], 'AGE':ph.iloc[0]['e_AGE'], 'DAP':ph.iloc[0]['e_DAP'], 'SUMDTT':ph.iloc[0]['e_SUMDTT']},
                        #'1': {'istage_old': 'Term Spklt', 'istage': 'Emergence', 'desc': 'Emergence to End of Juvenile', 'date':ph.iloc[0]['Sim Term Spklt'], 'DOY':ph.iloc[0]['ts_DOY'], 'AGE':ph.iloc[0]['ts_AGE'], 'DAP':ph.iloc[0]['ts_DAP'], 'SUMDTT':ph.iloc[0]['ts_SUMDTT']},
                        '1': {'istage_old': 'Term Spklt', 'istage': 'Emergence', 'desc': 'Emergence to End of Juvenile', 'date':ph.iloc[0]['Sim End Veg'], 'DOY':ph.iloc[0]['ev_DOY'], 'AGE':ph.iloc[0]['ev_AGE'], 'DAP':ph.iloc[0]['ev_DAP'], 'SUMDTT':ph.iloc[0]['ev_SUMDTT']},
                        '2': {'istage_old': 'End Veg', 'istage': 'End Juveni', 'desc': 'End of Juvenile to End of Vegetative growth', 'date':ph.iloc[0]['SimHeading'], 'DOY':ph.iloc[0]['h_DOY'], 'AGE':ph.iloc[0]['h_AGE'], 'DAP':ph.iloc[0]['h_DAP'], 'SUMDTT':ph.iloc[0]['h_SUMDTT']},
                        #'2': {'istage_old': 'End Veg', 'istage': 'End Juveni', 'desc': 'End of Juvenile to End of Vegetative growth', 'date':ph.iloc[0]['Sim End Veg'], 'DOY':ph.iloc[0]['ev_DOY'], 'AGE':ph.iloc[0]['ev_AGE'], 'DAP':ph.iloc[0]['ev_DAP'], 'SUMDTT':ph.iloc[0]['ev_SUMDTT']},
                        #'2.5': {'istage_old': 'Anthesis', 'istage': 'Anthesis', 'desc': 'Anthesis', 'date':ph.iloc[0]['SimAnthesis'], 'DOY':ph.iloc[0]['a_DOY'], 'AGE':ph.iloc[0]['a_AGE'], 'DAP':ph.iloc[0]['a_DAP'], 'SUMDTT':ph.iloc[0]['a_SUMDTT']},    
                        '3': {'istage_old': 'End Ear Gr', 'istage': 'End Veg', 'desc': 'End of Vegetative Growth to End of Ear Grow', 'date':ph.iloc[0]['Sim End Pan Gr'], 'DOY':ph.iloc[0]['epg_DOY'], 'AGE':ph.iloc[0]['epg_AGE'], 'DAP':ph.iloc[0]['epg_DAP'], 'SUMDTT':ph.iloc[0]['epg_SUMDTT']},
                        '4': {'istage_old': 'Beg Gr Fil', 'istage': 'End Ear Gr', 'desc': 'End of Ear Growth to Start of Grain Filling', 'date':ph.iloc[0]['Sim Beg Gr Fil'], 'DOY':ph.iloc[0]['bgf_DOY'], 'AGE':ph.iloc[0]['bgf_AGE'], 'DAP':ph.iloc[0]['bgf_DAP'], 'SUMDTT':ph.iloc[0]['bgf_SUMDTT']},
                        '5': {'istage_old': 'End Gr Fil', 'istage': 'Beg Gr Fil', 'desc': 'Start of Grain Filling to Maturity', 'date':ph.iloc[0]['SimMaturity'], 'DOY':ph.iloc[0]['m_DOY'], 'AGE':ph.iloc[0]['m_AGE'], 'DAP':ph.iloc[0]['m_DAP'], 'SUMDTT':ph.iloc[0]['m_SUMDTT']},
                        '6': {'istage_old': 'Harvest', 'istage': 'Maturity', 'desc': 'End Gr Fil', 'date':ph.iloc[0]['SimHarvest'], 'DOY':ph.iloc[0]['hv_DOY'], 'AGE':ph.iloc[0]['hv_AGE'], 'DAP':ph.iloc[0]['hv_DAP'], 'SUMDTT':ph.iloc[0]['hv_SUMDTT']}
                }
                print("RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE ")
                for k in growstages.keys():
                    print("{:4}   {:10} {:>10} {:>4} {:>6} {:>12} {:>12}".format(k, growstages[k]['istage_old'], 
                                                                                 growstages[k]['DAP'],
                                                                                 growstages[k]['DOY'], growstages[k]['AGE'],
                                                                                 growstages[k]['SUMDTT'], 
                                                                                 growstages[k]['date']))
                #
                if (output):
                    with open(f"{output}.{fmt}", "w") as f:
                        f.writelines("RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE \n")
                        for k in growstages.keys():
                            f.writelines("{:4}   {:10} {:>10} {:>4} {:>6} {:>12} {:>12}\n".format(k, growstages[k]['istage_old'], 
                                    growstages[k]['DAP'], growstages[k]['DOY'], growstages[k]['AGE'], 
                                    growstages[k]['SUMDTT'], growstages[k]['date']))
                return growstages
            elif (fmt=='csv'):
                if (output):
                    ph.to_csv(f"{output}.{fmt}")
            else:
                return ph  
    except Exception as err:
        print("No enough daily weather data to estimate phenology. Error: ",str(err))
      
    return res_arr

# ----------------------------------




