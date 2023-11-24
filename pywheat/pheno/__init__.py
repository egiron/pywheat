# coding=utf-8
#******************************************************************************
#
# Estimating Wheat phenological stages
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
import time
import math
import numpy as np
import pandas as pd
import datetime as dt

from ..utils import drawPhenology
from ..data import load_configfiles

def crown_temperatures(snow_depth=0, Tmin=None, Tmax=None):
    '''
        Crown temperatures are simulated according to the original routines in CERES-Wheat and the correspond 
        to air temperatures for non-freezing temperatures. The minimum and maximum crown temperatures (Tcmin and Tcmax) 
        are calculated according to the maximum and minimun air temperatures (Tmax and Tmin), respectively.

        Parameters:
            snow_depth (int): Snow depth in centimeters (cm). Default value is set to zero.
            Tmin (float): Minimum Temperature (°C)
            Tmax (float): Maximum Temperature (°C)

        Returns:
            Tcmin (float): Minimum Crown Temperature (°C)
            Tcmax (float): Maximum Crown Temperature (°C)
            Tcrown (float): Optimum Crown Temperature (°C)
        
    '''
    if (Tmin is None or Tmax is None):
        print("Please check out your inputs.")
        return
    
    Tcmax = None
    Tcmin = None

    def calc_CrownTemp(T, snow_depth=0):
        Tcrown = T
        snow_depth = min(snow_depth, 15)
        if (T < 0.0):
            Tcrown = 2.0 + T * (0.4 + 0.0018 * (snow_depth - 15)**2 )
        return Tcrown

    # Crown temperature for maximum development rate
    Tcmax = calc_CrownTemp(Tmax, snow_depth)
    # Crown temperature when snow is present and TMIN < 0.
    Tcmin = calc_CrownTemp(Tmin, snow_depth)

    Tcrown = (Tcmax + Tcmin) / 2 
    
    return Tcmax, Tcmin, Tcrown


# -------------
def thermal_time_calculation(m='CERES', snow_depth=0, Tmin=None, Tmax=None, Tbase=0, Topt=26, Ttop=34):
    '''
        The daily thermal time (daily_TT) or Growing degree days calculation

        It's calculated from the daily average of maximum and minimum crown temperatures, 
        and is adjusted by genetic and environments factors.

        Parameters:
            m (str): Name of the model. Default is 'CERES'. Options: CERES, NWHEAT, WHAPS
            snow_depth (int): Snow depth in centimeters (cm). Default value is set to zero.
            Tmin (float): Minimum Temperature (°C)
            Tmax (float): Maximum Temperature (°C)
            Tbase (float): Base temperature for development from ecotype database. Default 0°C
            Topt (float): Optimum temperature for development from species database. Default 26°C
            Ttop (float): Maximum temperature for development from species database. Default 34°C

        Returns:
            dTT (float): Thermal time or Growing degree days
        
    '''
    if (Tmin is None or Tmax is None):
        print("Check input parameters")
        return
    # Calculate Crown Temperatures
    Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=snow_depth, Tmin=Tmin, Tmax=Tmax)
    tcdif = Tcmax - Tcmin
    dTT = Tcrown - Tbase
    if (m=='CERES'):
        if (tcdif == 0): tcdif = 1.0
        if (Tcmax < Tbase):
            dTT = 0.0
        elif(Tcmax < Topt):
            if (Tcmin < Tbase):
                tcor = (Tcmax - Tbase) / tcdif
                dTT = (Tcmax - Tbase) / 2 * tcor
            else:
                dTT = Tcrown - Tbase
        elif(Tcmax < Ttop):
            if (Tcmin < Topt):
                tcor = (Tcmax - Topt) / tcdif
                # dTT = 13. * (1 + tcor) + Tcmin/2 * (1 - tcor)
                dTT = (Topt - Tbase) / 2 * (1 + tcor) + Tcmin/2 * (1 - tcor)
            else:
                dTT = Topt - Tbase
        else:
            if (Tcmin < Topt):
                tcor = (Tcmax - Ttop) / tcdif
                # dTT = (60 - Tcmax) * tcor + Topt * (1 - tcor)
                dTT = (Topt + Ttop - Tcmax) * tcor + Topt * (1 - tcor)
                tcor =  (Topt - Tcmin) / tcdif
                dTT = dTT * (1 - tcor) + (Tcmin + Topt) / 2 * tcor
            else:
                tcor = (Tcmax - Ttop) / tcdif
                dTT = (Topt + Ttop - Tcmax) * tcor + Topt * (1 - tcor)
        #
    #elif(m=='NWHEAT'):
    #    if ((Tcrown > 0) and (Tcrown <= 26)):
    #        dTT = Tcrown
    #    elif((Tcrown > 26) and (Tcrown <= 34)):
    #        dTT = (26/8) * (34 - Tcrown)
    #    elif((Tcrown <= 0) or (Tcrown > 34)):
    #        dTT = 0
        
    return round(dTT, 2)

# ---------
def day_length(DOY=1, lat=0.0, p=0.0):
    '''
        Length of the day for a specific site

        Day length is calculated from day of year (DOY), latitude and the civil twilight using 
        standard astronomical equations. Twilight is defined as the interval between sunrise or 
        sunset and the time whan the true center of the sun is 6° below the horizon.

        Parameters:
            DOY (int): Day of year
            lat (float): Latitude of the site in celsius degrees
            p (float):  Sun angle with the horizon. eg. p = 6.0 : civil twilight,
                        p = 0.0 : day starts / ends when sun is even with the horizon.
                        Default value p=0

        Returns: 
            daylength (float): A daylength for the specific site
        
    '''
    S1 = math.sin(lat * 0.0174533) # np.sin(np.deg2rad(lat))
    C1 = math.cos(lat * 0.0174533) # np.cos(np.deg2rad(lat))
    DEC = 0.4093 * math.sin( 0.0172 * (DOY - 82.2) )
    DLV = ( ( -S1 * math.sin(DEC) - 0.1047 ) / ( C1 * math.cos(DEC) ) )
    DLV = max(DLV,-0.87)
    TWILEN = 7.639 * math.acos(DLV)

    # Another way to calculate
    #latInRad = np.deg2rad(lat)
    #revolutionAngle = 0.2163108 + 2*np.arctan(0.9671396*np.tan(0.00860 *(dayOfYear - 186)))
    #declinationAngle = np.arcsin(0.39795*np.cos(revolutionAngle))
    #value = (np.sin(np.deg2rad(p)) + (np.sin(latInRad)*np.sin(declinationAngle))) / (np.cos(latInRad)*np.cos(declinationAngle))
    #if value <= -1.0: 
    #    return  0.0
    #if value >= 1.0: 
    #    return 24.0
    #else: 
    #    return 24 - (24/np.pi)*np.arccos(value)
        
    
    return TWILEN

'''
    Photoperiod

    The photoperiod (i.e. daylight hours) is a variable highly related to plant development, 
    particularly regulating the transition between vegetative and reproductive stages, 
    a transition that is typically characterized by flowering. This is an evolutionary 
    adaptation to ensure that seed production occurs during the right environmental conditions 
    to increase the survival rate and ensure the perpetuation of the specie.

    Because of its tilted angle and the consistent orbit of the Earth around the sun, 
    the theoretical photoperiod can be accurately estimated based on the day of the year and 
    latitude. Of course, the effective photoperiod observed at the Earth's surface may change
    depending on sky conditions.

    Parameters:
        DOY (int): Day of year
        lat (float): Latitude of the site in celsius degrees
        p (float):  Sun angle with the horizon. eg. p = 6.0 : civil twilight,
                    p = 0.0 : day starts / ends when sun is even with the horizon.
                    Default value p=0

    Usage:
        # Invoke function with scalars
        phi = 33.4;  # Latitude for consistency with notation in literature.
        doy = np.array([201]); # Day of the year. Julian calendar. Day from January 1.

        P = photoperiod(phi,doy,verbose=True)
        print('Photoperiod: ' + str(np.round(P[0],2)) + ' hours/day')

        # Multiple inputs call
        phi = 33.4;
        doy = np.arange(1,365);
        P = photoperiod(phi,doy)

        plt.figure(figsize=(8,6))
        plt.plot(doy,P)
        plt.title('Latitude:' + str(phi))
        plt.xlabel('Day of the year', size=14)
        plt.ylabel('Photoperiod (hours per day)', size=14)
        plt.show()


    Returns: 
        daylength (float): A daylength for the specific site
    
'''
# def photoperiod(phi,doy,verbose=False):
#     # Taken from https://github.com/soilwater/pynotes-agriscience
#     # Not used in calculation to avoid use of the numpy for future parallelization in GPUs
#     phi = np.radians(phi) # Convert to radians
#     light_intensity = 2.206 * 10**-3

#     C = np.sin(np.radians(23.44)) # sin of the obliquity of 23.44 degrees.
#     B = -4.76 - 1.03 * np.log(light_intensity) # Eq. [5]. Angle of the sun below the horizon. Civil twilight is -4.76 degrees.

#     # Calculations
#     alpha = np.radians(90 + B) # Eq. [6]. Value at sunrise and sunset.
#     M = 0.9856*doy - 3.251 # Eq. [4].
#     lmd = M + 1.916*np.sin(np.radians(M)) + 0.020*np.sin(np.radians(2*M)) + 282.565 # Eq. [3]. Lambda
#     delta = np.arcsin(C*np.sin(np.radians(lmd))) # Eq. [2].

#     # Defining sec(x) = 1/cos(x)
#     P = 2/15 * np.degrees( np.arccos( np.cos(alpha) * (1/np.cos(phi)) * (1/np.cos(delta)) - np.tan(phi) * np.tan(delta) ) ) # Eq. [1].

#     # Print results in order for each computation to match example in paper
#     if verbose:
#         print('Input latitude =', np.degrees(phi))
#         print('[Eq 5] B =', B)
#         print('[Eq 6] alpha =', np.degrees(alpha))
#         print('[Eq 4] M =', M[0])
#         print('[Eq 3] Lambda =', lmd[0])
#         print('[Eq 2] delta=', np.degrees(delta[0]))
#         print('[Eq 1] Daylength =', P[0])
#     return P


# ------------
def photoperiod_factor(P1D=3.675, day_length=20):
    '''
        Photoperiod factor 
        
        Phenology is affected by photoperiod between emergence and floral initiation, and 
        thermal time is affected by a photoperiod factor.

        Parameters:
            P1D (float): The sensitive to photoperiod (P1D) which is cultivar-specific. (1 - 6, low- high sensitive to day length)
            day_length (float): Day length in hours

        Returns:
            DF (float): A photoperiod factor
        
    '''
    DF = 1 - (0.002 * P1D) * (20 - day_length)**2
    return DF

def vernalization_factor(P1V=1.00, dV=50, ISTAGE=1):
    '''
        Calculation of vernalization factor.

        Phenology is affected by vernalization between emergence and floral initiation, and 
        thermal time is affected by a vernalization factor.

        Parameters:
            P1V (float): The sensitive to vernalization (P1V) which is cultivar-specific. 1 for spring type, 5 for winter type
            dV (float): The total vernalization. 

        Returns:
            VF (float): A vernalization factor
        
    '''
    # Set genetic coefficients to appropriate units
    #VSEN = params['P1V'] * 0.0054545 + 0.0003 
    # VF = 1 - VSEN * (50 - CUMVD)
    if (ISTAGE==1): #or ISTAGE==2
        VF = 1 - (0.0054545 * P1V  + 0.0003) * ( 50 - dV )
        VF = max(min(VF, 1.0), 0.0)
    else:
        VF = 1.0
    return VF


def vernalization(Tcrown, Tmin, Tmax, cumvd=0):
    '''
        Calculate damage to crop due to cold weather. 
        
        Vernalization is a response to relatively cold temperatures 
        in some species that must occur before reproductive growth will begin. 
        For wheat, temperature above zero to about 8°C seem to be the most effective 
        for vernalization (Ahrens & Loomis, 1963; Chujo, 1966).
        
        Vernalization affects phenology between emergence and floral initiation.
        Spring-type winter cereals have little sensitivity to vernalization, which is 
        the principal difference between them and the winter types.

        In the model, if the number of vernalization days (cumvd) is less than 10 and 
        the maximum temperature exceeds 30°C, then the number of vernalization days decreases 
        by 0.5 days per degree above 30°C. If cumvd is greater than 10, no devernalization is calculated. 

        Vernalization is simulated from daily average crown temperature (Tcrown), daily maximum (Tmax) and
        minimum (Tmin) temperatures using the original CEREES approach.

        Parameters:
            Tcrown (float): daily average crown temperature (°C)
            Tmin (float): daily average minimum temperature (°C)
            Tmax (float): daily average maximum temperature (°C)
            cumvd (float): the number of vernalization days of total vernalization
            
        Returns:
            dV (float): Vernalization
    '''
    # TODOs: Add threshold values as a global variables for VERN_TMIN, VERN_TMAX and CUMVD
    # Vernalization
    if( (Tmin < 15) and (Tmax > 0.0) ): #and Tmax <= 30 ISTAGE == 1 || ISTAGE == 9
        vd1 =  1.4 - 0.0778 * Tcrown
        # vd2 =  0.5 + 13.44 * ( Tcrown / ((Tmax - Tmin + 3)**2)) # wrong extract by APSIM-Wheat documentation
        vd2 =  0.5 + 13.44 / (Tmax - Tmin + 3)**2 * Tcrown # Extract by CERES Wheat 2.0 fortran code
        vd =  min(1.0, vd1, vd2)
        vd =  max(vd, 0.0)
        cumvd = cumvd + vd
    # Devernalization
    elif(Tmax > 30 and cumvd < 10): 
        cumvd = cumvd - 0.5 * (Tmax - 30)
        cumvd = max(cumvd, 0.0) 
        #cumvd = max(min(0.5 * (Tmax - 30), cumvd), 0.0)
    # ---------------------
    # VF = vernalization_factor(P1V=1.5, dV=20, ISTAGE=ISTAGE)
    #if (ISTAGE != 9):
    #    VF = 1.0 - P1V * (VREQ - cumvd) 
    #    if (VF <= 0.0):
    #        VF = 0.0
    #elif (VF > 1.0):
    #    VF = 1.0
    # ---------------------
    
    return cumvd


# ------------
def snow_fall(Tmax, Rain):
    '''
        Cold weather handling routine (extracted from watbal subroutine)

        Parameters:
            Rain (float): Precipitation depth for current day (mm)
            Tmax (float): Maximum daily temperature (°C)
        
        Returns:
            snow_melt (float): Daily Snowmelt (mm/d)
            snow (float): Snow accumulation (mm)
            water_available (float): Water available for infiltration or runoff (rainfall plus irrigation) (mm/d)
                
    '''
    snow = 0.0
    snow_melt = 0.0
    water_available = 0.0
    if (Tmax > 1.0):
        snow_melt = Tmax + Rain * 0.4
        if (snow_melt > snow):
            snow_melt = snow
        snow = snow - snow_melt
        water_available = Rain + snow_melt
    else:
        snow = snow + Rain
    #
    return snow, snow_melt, water_available

# ====================================================
# Wheat phenological stages
# ====================================================
def determine_phenology_stage(initparams=None, weather=None, dispDates=True, dispFigPhenology=False, verbose=False):
    '''
        Estimate Wheat phenological stages using CERES-Wheat model
        
        Warning: Deprecated.
                This function was depreciated on Oct 22, 2023. 
                Please use an updated version named `determine_phenology_stages`

        Parameters:
            initparams (dict): A dictionary with initial parameters
            weather (object): A table or dataframe with weather data for the site
            dispDates (bool): Display results in text format. Default is True
            dispFigPhenology (bool): Display a figure with the phenological phases. Default is False
            verbose (bool): Display comments during the processes. Default is False
        
        Attributes:
            TT_TBASE (float): Base temperature for estimate Thermal time. Default 0.0
            TT_TEMPERATURE_OPTIMUM (float): Thermal time optimum temperature. Default 26
            TT_TEMPERATURE_MAXIMUM (float): Thermal time maximum temperature. Default 34
            CIVIL_TWILIGHT (float): Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0
            HI (float): Hardiness Index. Default 0.0 
            SNOW (float): Snow fall. Default 0.0
            SDEPTH (float): Sowing depth in cm. Default 3.0 cm
            GDDE (float): Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.
            DSGFT (float): GDD from End Ear Growth to Start Grain Filling period. Default 200 degree-days
            VREQ  (float): Vernalization required for max.development rate (VDays). Default 505 degree-days
            PHINT (float): Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. Default 95.0 degree-days
            P1V (float): Development genetic coefficients, vernalization. 1 for spring type, 5 for winter type. Default 4.85
            P1D (float): Development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length). Default 3.675
            P5 (float): Grain filling degree days. Old value was divided by 10. Default 500 degree-days.
            P6 (float): Approximate the thermal time from physiological maturity to harvest. Default 250.
            DAYS_GERMIMATION_LIMIT (float): Threshold for days to germination. Default 40
            TT_EMERGENCE_LIMIT (float): Threshold for thermal time to emergence. Default 300
            TT_TDU_LIMIT (float): Threshold for thermal development units (TDU). Default 400 
            
        Returns:
            growstages (dict): A dictionary with all phenological stages and addtional useful information
            
    '''
    if (initparams is None):
        print("Please check out the input parameters")
        return
    if (weather is None):
        print("Weather data is not available")
        return
    
    # Initialization of variables 
    params = dict(
        sowing_date = "", # Sowing date in YYYY-MM-DD
        latitude = -99.0, # Latitude of the site
        TT_TBASE = 0.0, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = 26, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = 34, # Thermal time maximum temperature
        CIVIL_TWILIGHT = 0.0, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        HI = 0.0, # Hardiness Index
        SNOW = 0, # Snow fall
        SDEPTH = 3.0, # Sowing depth in cm
        GDDE = 6.2, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = 200, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = 505.0, # Vernalization required for max.development rate (VDays)
        PHINT = 95.0, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = 1.0, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = 3.675, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = 500, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = 250, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = 40, # threshold for days to germination
        TT_EMERGENCE_LIMIT = 300, # threshold for thermal time to emergence
        TT_TDU_LIMIT = 400, # threshold for thermal development units (TDU)

    )
    if (initparams is not None):
        params = {**params, **initparams}
    
    # ---------------------
    # GDD limits
    # ---------------------
    P2 = params['PHINT'] * 3
    P3 = params['PHINT'] * 2
    P4 = params['DSGFT'] #200 # APSIM-Wheat = 120 # GDD from End Ear Growth to Start Grain Filling period
    P5 = params['P5'] #430 + params['P5'] * 20
    P6 = params['P5'] #250
    #P9 = 40 + params['GDDE'] * params['SDEPTH'] 
    
    growstages = {
            '7': {'istage_old': 'Sowing', 'istage': 'Fallow', 'desc': 'No crop present to Sowing', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '8': {'istage_old': 'Germinate', 'istage': 'Sowing', 'desc': 'Sowing to Germination', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '9': {'istage_old': 'Emergence', 'istage': 'Germinate', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '1': {'istage_old': 'Term Spklt', 'istage': 'Emergence', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '2': {'istage_old': 'End Veg', 'istage': 'End Juveni', 'desc': 'End of Juvenile to End of Vegetative growth', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '3': {'istage_old': 'End Ear Gr', 'istage': 'End Veg', 'desc': 'End of Vegetative Growth to End of Ear Grow', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '4': {'istage_old': 'Beg Gr Fil', 'istage': 'End Ear Gr', 'desc': 'End of Ear Growth to Start of Grain Filling', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '5': {'istage_old': 'End Gr Fil', 'istage': 'Beg Gr Fil', 'desc': 'Start of Grain Filling to Maturity', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '6': {'istage_old': 'Harvest', 'istage': 'Maturity', 'desc': 'End Gr Fil', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''}
    }
    
    class StageFailed(Exception):
        def __init__(self, m, istage, err):
            self.message = m
            self.istage = istage
            self.err = err
        def __str__(self):
            return self.message + f" Stage ({self.istage}) - " + f"Error: {self.err}"

    # --------------------------------------------------------------------------
    # DETERMINE SOWING DATE
    # --------------------------------------------------------------------------
    ISTAGE = 7
    try:
        SOWING_DATE = pd.to_datetime(str(params['sowing_date']), format='%Y-%m-%d' )
        DOY = pd.to_datetime(SOWING_DATE).dayofyear

        growstages[f'{ISTAGE}']['date'] = str(SOWING_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(DOY)
        growstages[f'{ISTAGE}']['AGE'] = 0
        growstages[f'{ISTAGE}']['SUMDTT'] = 0
        growstages[f'{ISTAGE}']['DAP'] = 0
        #print("Sowing date:", SOWING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem initializing the determination of phenological stage. Please check your input parameters such as sowing date or latitude of the site", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
        

    # --------------------------------------------------------------------------
    # DETERMINE GERMINATION  DATE
    # --------------------------------------------------------------------------
    ISTAGE = 8
    try:
        SUMDTT = 0.0
        #VF = 0.0
        DAP = 0
        ndays = 1 # Seed germination is a rapid process and is assumed to occur in one day
        w = weather[(weather['DATE']==(SOWING_DATE + pd.DateOffset(days=ndays)) )].reset_index(drop=True)
        GERMINATION_DATE = ''
        Tmin = float(w.iloc[ndays-1]['TMIN'])
        Tmax = float(w.iloc[ndays-1]['TMAX'])
        # Thermal time
        DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                       Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                       Ttop=params['TT_TEMPERATURE_MAXIMUM'])
        SUMDTT = SUMDTT + DTT
        GERMINATION_DATE = w.iloc[ndays-1]['DATE']
        CROP_AGE = str(GERMINATION_DATE - SOWING_DATE).replace(' days 00:00:00','')
        DAP = DAP + int(CROP_AGE)
        growstages[f'{ISTAGE}']['date'] = str(GERMINATION_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(GERMINATION_DATE.dayofyear)
        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
        growstages[f'{ISTAGE}']['DAP'] = DAP

        #print("Germination date:", GERMINATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining germination date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return


    # --------------------------------------------------------------------------
    # DETERMINE SEEDLING EMERGENCE DATE
    # --------------------------------------------------------------------------
    ISTAGE = 9
    try:
        P9 = 40 + params['GDDE'] * params['SDEPTH'] #Default values
        SUMDTT = 0.0
        #print("Growing degree days from germination to emergence (P9): ",P9) 
        # The crop will die if germination has not occurred before a certain period (eg. 40 days)
        w = weather[weather['DATE']>=GERMINATION_DATE].reset_index(drop=True)
        EMERGENCE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            SUMDTT = SUMDTT + DTT

            if (SUMDTT >= P9 or SUMDTT > params['TT_EMERGENCE_LIMIT']):
                EMERGENCE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(EMERGENCE_DATE - GERMINATION_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(EMERGENCE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(EMERGENCE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(EMERGENCE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break

        #print("Emergence date: ", EMERGENCE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining emergence date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------------------
    # DETERMINE DURATION OF VEGETATIVE PAHSE (END JUVENILE DATE - END OF VEGETATION GROWTH
    # --------------------------------------------------------------------------------------
    ISTAGE = 1
    try: 
        isVernalization = True
        SUMDTT = SUMDTT - P9 
        CUMVD = 0
        TDU = 0
        DF = 0.001
        shoot_lag = 40 # Assumed to be around 40 °C d
        shoot_rate = 1.5 # 1.5 °C d per mm. dDerived from studies where thermal time to emergence was measured and where sowing depth was known
        sowing_depth = params['SDEPTH'] * 10.0 # mm or 3cm as CERES

        T_emer = shoot_lag + shoot_rate * sowing_depth
        #print("Thermal time to emergence date: {} °C d".format( T_emer))
        #print("Thermal time to emergence date in CERES (P9): {} °C d".format(P9))
        TT_emergence = min(T_emer, P9)

        w = weather[weather['DATE']>=EMERGENCE_DATE].reset_index(drop=True)
        END_JUVENILE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            if (isVernalization is True):
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                if (CUMVD < params['VREQ']):
                    VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                    if (VF < 0.3):
                        TDU = TDU + DTT * min(VF, DF)
                    else:
                        DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                        TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                        DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                        TDU = TDU + DTT * min(VF, DF)
                    SUMDTT = TDU
                else:
                    isVernalization = False
            else:
                SUMDTT = SUMDTT + DTT

            if (SUMDTT > P9 or SUMDTT > TT_emergence): # when reached the lower TT
                END_JUVENILE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(END_JUVENILE_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(END_JUVENILE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(END_JUVENILE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(END_JUVENILE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break
            if (DTT > params['TT_EMERGENCE_LIMIT']): # TT_EMERGENCE_LIMIT = 300,
                # The crop will die if germination has not occurred before a certain period (eg. 40 days or 300oC d)
                print("The crop died because emergence has not occurred before {} degree-days".format(params['TT_EMERGENCE_LIMIT']))

        #print("End Juvenile date: ", END_JUVENILE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of juvenile date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------
    # DETERMINE END VEGETATION DATE - End of Juvenile to End of Vegetative growth
    # --------------------------------------------------------------------------
    ISTAGE = 1 # <- Note: this must continue with 1 as previous stage (Term Spklt = Emergence to End of Juvenile + End of Juvenile to End of Vegetative growth)
    try:
        isVernalization = True
        VF = 1.0
        w = weather[weather['DATE']>=END_JUVENILE_DATE].reset_index(drop=True)
        END_VEGETATION_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                if (isVernalization is True):
                    Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                    CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                    if (CUMVD < params['VREQ']):
                        VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                        if (VF < 0.3):
                            TDU = TDU + DTT * min(VF, DF)
                        else:
                            DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                            TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                            DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                            TDU = TDU + DTT * min(VF, DF)
                        SUMDTT = TDU
                    else:
                        isVernalization = False
                else:
                    SUMDTT = SUMDTT + DTT

                # When this reduced thermal time accumulation (TDU) reaches 
                # 400 degree days, Stage 1 development ends
                if (SUMDTT > (params['TT_TDU_LIMIT'] * (params['PHINT'] / 95.0)) ):
                    END_VEGETATION_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_VEGETATION_DATE - END_JUVENILE_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    # Sum of the two phases
                    CROP_AGE_2 = str(END_VEGETATION_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE_2) # Sum of the last two phases
                    growstages[f'{ISTAGE}']['date'] = str(END_VEGETATION_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_VEGETATION_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("End of Juvenile: Thermal time reached at days duration ", i+1,
                    #          str(END_VEGETATION_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                    break
        else:
            print("Error reading weather data for vegetation phase")

        # print("End of Vegeation Growth ", END_VEGETATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of vegetation growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF EAR GROWTH - End of Vegetative Growth to End of Ear Grow (End leaf growth)
    #-----------------------------------------------------------------------------------------------
    ISTAGE = 2 # Terminal spikelet initiation to the end of leaf growth - CERES Stage 2
    try:
        SUMDTT = 0.0
        P2 = params['PHINT'] * 3

        w = weather[weather['DATE']>=END_VEGETATION_DATE].reset_index(drop=True)
        END_OF_EAR_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P2):
                    END_OF_EAR_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_EAR_GROWTH_DATE - END_VEGETATION_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_EAR_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_EAR_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_EAR_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Ear growth",END_OF_EAR_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of ear growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF PANNICLE GROWTH - End pannicle growth - End of Ear Growth to Start of Grain Filling
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 3 # Preanthesis ear growth - CERES Stage 3.
    try:
        SUMDTT = 0.0 #SUMDTT - P2
        P3 = params['PHINT'] * 2
        #TBASE=0.0

        w = weather[weather['DATE']>END_OF_EAR_GROWTH_DATE].reset_index(drop=True)
        END_OF_PANNICLE_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P3):
                    END_OF_PANNICLE_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_PANNICLE_GROWTH_DATE - END_OF_EAR_GROWTH_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_PANNICLE_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_PANNICLE_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_PANNICLE_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Pre-Anthesis Ear growth",END_OF_PANNICLE_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of pre-anthesis earh growth date (end of pannicle growth date).", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE BEGIN GRAIN FILLING - Grain fill - Start of Grain Filling to Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 4 # Preanthesis ear growth to the beginning of grain filling - CERES Stage 4.
    try: 
        P4 = params['DSGFT'] #200 GDD # APSIM-Wheat = 120
        SUMDTT = 0.0 #SUMDTT - P3

        w = weather[weather['DATE']>=END_OF_PANNICLE_GROWTH_DATE].reset_index(drop=True)
        BEGIN_GRAIN_FILLING_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT
                if (SUMDTT >= P4):
                    BEGIN_GRAIN_FILLING_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(BEGIN_GRAIN_FILLING_DATE - END_OF_PANNICLE_GROWTH_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(BEGIN_GRAIN_FILLING_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(BEGIN_GRAIN_FILLING_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(BEGIN_GRAIN_FILLING_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("Begining of Grain fill",BEGIN_GRAIN_FILLING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining begin of grain fill date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END GRAIN FILLING - Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 5
    try:
        SUMDTT = 0.0 #SUMDTT - P4
        #P5 = 430 + params['P5'] * 20 # P5 = (0.05 X TT_Maturity) - 21.5. ~500 degree-days
        P5 = params['P5'] # 400 + 5.0 * 20  

        w = weather[weather['DATE']>=BEGIN_GRAIN_FILLING_DATE].reset_index(drop=True)
        END_GRAIN_FILLING_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT
                if (SUMDTT >= P5):
                    END_GRAIN_FILLING_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_GRAIN_FILLING_DATE - BEGIN_GRAIN_FILLING_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_GRAIN_FILLING_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_GRAIN_FILLING_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_GRAIN_FILLING_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Grain filling",END_GRAIN_FILLING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of grain fill date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE HARVEST - Harvest - End of Grain Filling or Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 6  # Physiological maturity to harvest - CERES Stage 6.
    try: 
        SUMDTT = 0.0
        estimateHarvest = True
        P6 = params['P6']

        if (estimateHarvest is False):
            HARVEST = END_GRAIN_FILLING_DATE
            growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
            growstages[f'{ISTAGE}']['date'] = str(HARVEST).split(' ')[0]
            growstages[f'{ISTAGE}']['DOY'] = int(HARVEST.dayofyear)
            growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
            growstages[f'{ISTAGE}']['DAP'] = DAP
        else:
            # A value of 250 degree-days can be used to approximate the thermal time from physiological maturity to harvest
            w = weather[weather['DATE']>=END_GRAIN_FILLING_DATE].reset_index(drop=True)
            HARVEST = ''
            if (len(w)>0):
                for i in range(len(w)):
                    Tmin = float(w.iloc[i]['TMIN'])
                    Tmax = float(w.iloc[i]['TMAX'])
                    # Thermal time
                    DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                                   Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                                   Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                    SUMDTT = SUMDTT + DTT
                    if (SUMDTT >= P6):
                        HARVEST = w.iloc[i]['DATE']
                        CROP_AGE = str(HARVEST - END_GRAIN_FILLING_DATE).replace(' days 00:00:00','')
                        DAP = DAP + int(CROP_AGE)
                        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                        growstages[f'{ISTAGE}']['date'] = str(HARVEST).split(' ')[0]
                        growstages[f'{ISTAGE}']['DOY'] = int(HARVEST.dayofyear)
                        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                        growstages[f'{ISTAGE}']['DAP'] = DAP
                        #if (verbose is True):
                        #    print("Thermal time reached at days duration ", i+1, str(HARVEST), 
                        #          CROP_AGE, DAP, round(SUMDTT, 1))
                        break
    except Exception as err:
        try:
            raise StageFailed("Problem determining physiological maturity to harvest date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
        
    # ---------------------
    if (dispDates is True):
        try:
            print("RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE ")
            for k in growstages.keys():
                print("{:4}   {:10} {:>10} {:>4} {:>6} {:>12} {:>12}".format(k, growstages[k]['istage_old'], 
                                                                             growstages[k]['DAP'],
                                                                             growstages[k]['DOY'], growstages[k]['AGE'],
                                                                             growstages[k]['SUMDTT'], 
                                                                             growstages[k]['date']))
        except Exception as err:
            try:
                raise StageFailed("Problem displaying results.", -99, err)
            except StageFailed as x:
                print(x)
                return
            
    if (dispFigPhenology is True):
        try:
            drawPhenology(gs=growstages, title='Phenological growth phases of Wheat', dpi=150,
                         dispPlants=True, topDAPLabel=True, timeSpanLabel=True, topNameStageLabel=True,
                         topNameStageLabelOpt=True, copyrightLabel=True, 
                         saveFig=False, showFig=True, path_to_save_results='./', 
                         fname='Fig_1_Phenological_Phases_Wheat', fmt='jpg')
        except Exception as err:
            try:
                raise StageFailed("Problem displaying figure of phenological stages.", -99, err)
            except StageFailed as x:
                print(x)
                return
    # ---------------------
    return growstages

# ==================================

# ==========================================================================
# Wheat phenological stages - updated version calibrated with IWIN datasets
# ==========================================================================
def determine_phenology_stages(config=None, initparams=None, useDefault=True, 
                               dispDates=True, dispFigPhenology=False, verbose=False):
    '''
        Estimate Wheat phenological stages using an improved PyWheat model calibrated 
        with IWIN datasets (ESWYT, IDYN, HTWYT and SAWYT nurseries)

        Parameters:
            config (dict): A dictionary with configuration parameters
            initparams (dict): A dictionary with initial parameters
            dispDates (bool): Display results in text format. Default is True
            dispFigPhenology (bool): Display a figure with the phenological phases. Default is False
            verbose (bool): Display comments during the processes. Default is False
        
        Attributes:
            weather (object): A table or dataframe with weather data for the site
            TT_TBASE (float): Base temperature for estimate Thermal time. Default 0.0
            TT_TEMPERATURE_OPTIMUM (float): Thermal time optimum temperature. Default 26
            TT_TEMPERATURE_MAXIMUM (float): Thermal time maximum temperature. Default 34
            CIVIL_TWILIGHT (float): Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0
            HI (float): Hardiness Index. Default 0.0 
            SNOW (float): Snow fall. Default 0.0
            SDEPTH (float): Sowing depth in cm. Default 3.0 cm
            GDDE (float): Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.
            DSGFT (float): GDD from End Ear Growth to Start Grain Filling period. Default 200 degree-days
            VREQ  (float): Vernalization required for max.development rate (VDays). Default 505 degree-days
            PHINT (float): Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. Default 95.0 degree-days
            P1V (float): Days,optimum vernalizing temperature,required for vernalization. Development genetic coefficients, vernalization. 1 for spring type, 5 for winter type. Default 1.00
            P1D (float): Photoperiod response (% reduction in rate/10 h drop in pp). Development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length). Default 3.675  #75 DSSAT
            P5 (float): Grain filling degree days. Old value was divided by 10. Default 500 degree-days.
            P6 (float): Approximate the thermal time from physiological maturity to harvest. Default 250.
            DAYS_GERMIMATION_LIMIT (float): Threshold for days to germination. Default 40
            TT_EMERGENCE_LIMIT (int): Threshold for thermal time to emergence. Default 400 degree-days
            TT_TDU_LIMIT (float): Threshold for thermal development units (TDU). Default 400  degree-days
            ADAH (int): Number of days after heading. A threshold used for anthesis date after planting. Default is 6 days after heading.
            p5_steps (float): Step to increase or reduce the P5 parameters. Default 1.0
            maxP5 (float): Threshold for the maximum value of P5 to reach maturity date. Default 1000
            
        Returns:
            growstages (dict): A dictionary with all phenological stages and addtional useful information
            
    '''
    if (config is None):
        config = load_configfiles()
    if (initparams is None):
        print("Please check out the input parameters")
        return
    
    # Initialization of variables 
    params = dict(
        weather = None, # Weather data of the site
        sowing_date = "", # Sowing date in YYYY-MM-DD
        latitude = -90.0, # Latitude of the site
        longitude = -180.0, # Longitude of the site
        genotype = "", # Name of the grand parent in IWIN pedigrees database 
        TT_TBASE = 0.0, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = 26, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = 34, # Thermal time maximum temperature
        CIVIL_TWILIGHT = 0.0, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        HI = 0.0, # Hardiness Index
        SNOW = 0, # Snow fall
        SDEPTH = 3.0, # Sowing depth in cm
        GDDE = 6.2, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = 200, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = 505.0, # Vernalization required for max.development rate (VDays)
        PHINT = 95.0, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = 1.0, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = 3.675, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = 500, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = 250, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = 40, # threshold for days to germination
        TT_EMERGENCE_LIMIT = 500, # threshold for thermal time to emergence
        TT_TDU_LIMIT = 1000, # threshold for thermal development units (TDU)
        ADAH = 6, # threshold for anthesis date after planting. This is a 6 days after heading.
        bruteforce = False,
        brute_params = {
            "obsEmergenceDAP": None, # Observed days after planting to emergence.
            "obsHeadingDAP": None, # Observed days after planting to heading.
            "obsAnthesisDAP": None, # Observed days after planting to Anthesis.
            "obsMaturityDAP": None, # Observed days after planting to Maturity.
            "max_tries": 300, # Number of maximum tries to find the best value
            "error_lim": 0.5, # Threshold to classify the observation as a good or bad
            "gdde_steps": 1.0, # Step to increase or reduce the GDDE parameters. Default 1.0
            "maxGDDE": 50, #Threshold for the maximum value of GDDE to reach emergence date
            "phint_steps": 1.0, # Step to increase or reduce the PHINT parameters. Default 1.0
            "maxPHINT": 150, #Threshold for the maximum value of PHINT to reach heading date
            "adap_steps": 1, #Step to increase or reduce the ADAH parameters. Default 1
            "maxADAP": 10, #Threshold for the maximum value of ADAH to reach anthesis date.
            "p5_steps": 1, #Step to increase or reduce the P5 parameters. Default 1.0
            "maxP5": 2000 #Threshold for the maximum value of P5 to reach anthesis date.
        }

    )
    if (initparams is not None):
        params = {**params, **initparams}
    
    # Validate
    if (params['sowing_date']=="" or params['sowing_date'] is None):
        print("Sowing date not defined")
        return
    if (params['latitude']==-90.0 or params['latitude'] is None):
        print("Problem with location of the site. Check the geographic coordinates.")
        return
    if (params['weather'] is None):
        print("Weather data is not available")
        return
    else:
        weather = params['weather']
    if ((params['bruteforce'] is True )
        and (params['brute_params']["obsEmergenceDAP"] is None and params['brute_params']["obsHeadingDAP"] is None 
             and params['brute_params']["obsAnthesisDAP"] is None and params['brute_params']["obsMaturityDAP"] is None
            )):
        print("Parameters to run brute force algorithm are not defined yet.")
        return
    
    # ---------------------
    # GDD limits
    # ---------------------
    P2 = params['PHINT'] * 3
    P3 = params['PHINT'] * 2
    P4 = params['DSGFT'] #200 # APSIM-Wheat = 120 # GDD from End Ear Growth to Start Grain Filling period
    P5 = params['P5'] #430 + params['P5'] * 20 # DSSAT v4.8
    #P6 = params['P5'] #250
    #P9 = 40 + params['GDDE'] * params['SDEPTH'] 
    
    growstages = {
            '7': {'istage_old': 'Sowing', 'istage': 'Fallow', 'desc': 'No crop present to Sowing', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '8': {'istage_old': 'Germinate', 'istage': 'Sowing', 'desc': 'Sowing to Germination', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '9': {'istage_old': 'Emergence', 'istage': 'Germinate', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '1': {'istage_old': 'Term Spklt', 'istage': 'Emergence', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '2': {'istage_old': 'End Veg', 'istage': 'End Juveni', 'desc': 'End of Juvenile to End of Vegetative growth', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '2.5': {'istage_old': 'Anthesis', 'istage': 'Anthesis', 'desc': 'Anthesis', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '3': {'istage_old': 'End Ear Gr', 'istage': 'End Veg', 'desc': 'End of Vegetative Growth to End of Ear Grow', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '4': {'istage_old': 'Beg Gr Fil', 'istage': 'End Ear Gr', 'desc': 'End of Ear Growth to Start of Grain Filling', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '5': {'istage_old': 'End Gr Fil', 'istage': 'Beg Gr Fil', 'desc': 'Start of Grain Filling to Maturity', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''},
            '6': {'istage_old': 'Harvest', 'istage': 'Maturity', 'desc': 'End Gr Fil', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':'', 'status':''}
    }
    
    class StageFailed(Exception):
        def __init__(self, m, istage, err):
            self.message = m
            self.istage = istage
            self.err = err
        def __str__(self):
            return self.message + f" Stage ({self.istage}) - " + f"Error: {self.err}"

    # --------------------------------------------------------------------------
    # DETERMINE SOWING DATE
    # --------------------------------------------------------------------------
    ISTAGE = 7
    try:
        SOWING_DATE = pd.to_datetime(str(params['sowing_date']), format='%Y-%m-%d' )
        DOY = pd.to_datetime(SOWING_DATE).dayofyear

        growstages[f'{ISTAGE}']['date'] = str(SOWING_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(DOY)
        growstages[f'{ISTAGE}']['AGE'] = 0
        growstages[f'{ISTAGE}']['SUMDTT'] = 0
        growstages[f'{ISTAGE}']['DAP'] = 0
        growstages[f'{ISTAGE}']['status'] = 1
        #print("Sowing date:", SOWING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem initializing the determination of phenological stage. Please check your input parameters such as sowing date or latitude of the site", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    
    # --------------------------------------------------------------------------
    # DETERMINE GERMINATION  DATE
    # --------------------------------------------------------------------------
    ISTAGE = 8
    try:
        SUMDTT = 0.0
        #VF = 0.0
        DAP = 0
        ndays = 1 # Seed germination is a rapid process and is assumed to occur in one day
        w = weather[(weather['DATE']==(SOWING_DATE + pd.DateOffset(days=ndays)) )].reset_index(drop=True)
        GERMINATION_DATE = ''
        Tmin = float(w.iloc[ndays-1]['TMIN'])
        Tmax = float(w.iloc[ndays-1]['TMAX'])
        # Thermal time
        DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                       Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                       Ttop=params['TT_TEMPERATURE_MAXIMUM'])
        SUMDTT = SUMDTT + DTT
        GERMINATION_DATE = w.iloc[ndays-1]['DATE']
        CROP_AGE = str(GERMINATION_DATE - SOWING_DATE).replace(' days 00:00:00','')
        DAP = DAP + int(CROP_AGE)
        growstages[f'{ISTAGE}']['date'] = str(GERMINATION_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(GERMINATION_DATE.dayofyear)
        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
        growstages[f'{ISTAGE}']['DAP'] = DAP
        growstages[f'{ISTAGE}']['status'] = 1

        #print("Germination date:", GERMINATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining germination date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return


    # --------------------------------------------------------------------------
    # DETERMINE SEEDLING EMERGENCE DATE
    # --------------------------------------------------------------------------
    ISTAGE = 9
    P9 = 40 + params['GDDE'] * params['SDEPTH'] #Default values defined by user or by CERES-Wheat
    try:
        # Get GDDE and SDEPTH parameters
        e_status = 1
        if (params['bruteforce']==True and params['brute_params']!={} and 
            params['brute_params']["obsEmergenceDAP"] is not None):
            # using brute force algorithms
            e_growstages, params, e_status = estimate_emergence_by_bruteforce(params)
            #params = { **params, **e_params}
        elif ( useDefault is True and params['longitude']!=-180.0 and params['longitude']!='' 
              and params['longitude'] is not None):
            ## using geographic coordinates and month to extract values by linear regression
            params = estimate_emergence_by_default(config, params, GERMINATION_DATE)
        elif (params['longitude']!=-180.0 and params['longitude']!='' and params['longitude'] is not None):
            ## using geographic coordinates
            params = estimate_emergence_by_coords(config, params)
        elif (params['genotype']!="" and params['genotype'] is not None):
            # using cultivar or genotype name
            params = estimate_emergence_by_cultivar(config, params, GERMINATION_DATE)
        
        #
        P9 = 40 + params['GDDE'] * params['SDEPTH']
        SUMDTT = 0.0
        #print("Growing degree days from germination to emergence (P9): ",P9) 
        # The crop will die if germination has not occurred before a certain period (eg. 40 days)
        
        EMERGENCE_DATE = ''
        w = weather[weather['DATE']>=GERMINATION_DATE].reset_index(drop=True)
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            SUMDTT = SUMDTT + DTT

            if (SUMDTT >= P9 or SUMDTT > params['TT_EMERGENCE_LIMIT']):
                EMERGENCE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(EMERGENCE_DATE - GERMINATION_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(EMERGENCE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(EMERGENCE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                growstages[f'{ISTAGE}']['status'] = e_status
                #print("Thermal time reached at DAP ", i+1, str(EMERGENCE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break

        #print("Emergence date: ", EMERGENCE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining emergence date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    
    # --------------------------------------------------------------------------------------
    # DETERMINE DURATION OF VEGETATIVE PAHSE (END JUVENILE DATE - END OF VEGETATION GROWTH
    # --------------------------------------------------------------------------------------
    ISTAGE = 1
    # Get SNOW, VREQ, P1V, P1D,  parameters using geographic coordinates
    h_status = 1
    if (params['bruteforce']==True and params['brute_params']!={} and 
        params['brute_params']["obsHeadingDAP"] is not None):
        # using brute force algorithms
        h_growstages, params, h_status = estimate_heading_by_bruteforce(params)
        #params = { **params, **h_params}
    elif ( useDefault is True and params['longitude']!=-180.0 and params['longitude']!='' 
          and params['longitude'] is not None):
        ## using geographic coordinates and month to extract values by linear regression
        params = estimate_heading_by_default(config, params, EMERGENCE_DATE)
    elif (params['longitude']!=-180.0 and params['longitude']!='' and params['longitude'] is not None):
        ## using geographic coordinates
        params = estimate_heading_by_coords(config, params)
    elif (params['genotype']!="" and params['genotype'] is not None):
        # using cultivar or genotype name
        params = estimate_heading_by_cultivar(config, params, EMERGENCE_DATE)
    
    try: 
        isVernalization = True
        SUMDTT = SUMDTT - P9 
        CUMVD = 0
        TDU = 0
        DF = 0.001
        shoot_lag = 40 # Assumed to be around 40 °C d
        shoot_rate = 1.5 # 1.5 °C d per mm. dDerived from studies where thermal time to emergence was measured and where sowing depth was known
        sowing_depth = params['SDEPTH'] * 10.0 # mm or 3cm as CERES

        T_emer = shoot_lag + shoot_rate * sowing_depth
        #print("Thermal time to emergence date: {} °C d".format( T_emer))
        #print("Thermal time to emergence date in CERES (P9): {} °C d".format(P9))
        TT_emergence = min(T_emer, P9)

        w = weather[weather['DATE']>=EMERGENCE_DATE].reset_index(drop=True)
        END_JUVENILE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            if (isVernalization is True):
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                if (CUMVD < params['VREQ']):
                    VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                    if (VF < 0.3):
                        TDU = TDU + DTT * min(VF, DF)
                    else:
                        DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                        TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                        DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                        TDU = TDU + DTT * min(VF, DF)
                    SUMDTT = TDU
                else:
                    isVernalization = False
            else:
                SUMDTT = SUMDTT + DTT

            if (SUMDTT > P9 or SUMDTT > TT_emergence): # when reached the lower TT
                END_JUVENILE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(END_JUVENILE_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(END_JUVENILE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(END_JUVENILE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                growstages[f'{ISTAGE}']['status'] = h_status
                #print("Thermal time reached at DAP ", i+1, str(END_JUVENILE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break
            if (DTT > params['TT_EMERGENCE_LIMIT']): # TT_EMERGENCE_LIMIT = 300,
                # The crop will die if germination has not occurred before a certain period (eg. 40 days or 300oC d)
                print("The crop died because emergence has not occurred before {} degree-days".format(params['TT_EMERGENCE_LIMIT']))

        #print("End Juvenile date: ", END_JUVENILE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of juvenile date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------
    # DETERMINE END VEGETATION DATE - End of Juvenile to End of Vegetative growth
    # --------------------------------------------------------------------------
    ISTAGE = 1 # <- Note: this must continue with 1 as previous stage (Term Spklt = Emergence to End of Juvenile + End of Juvenile to End of Vegetative growth)
    try:
        isVernalization = True
        VF = 1.0
        w = weather[weather['DATE']>=END_JUVENILE_DATE].reset_index(drop=True)
        END_VEGETATION_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                if (isVernalization is True):
                    Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                    CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                    if (CUMVD < params['VREQ']):
                        VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                        if (VF < 0.3):
                            TDU = TDU + DTT * min(VF, DF)
                        else:
                            DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                            TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                            DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                            TDU = TDU + DTT * min(VF, DF)
                        SUMDTT = TDU
                    else:
                        isVernalization = False
                else:
                    SUMDTT = SUMDTT + DTT

                # When this reduced thermal time accumulation (TDU) reaches 
                # 400 degree days, Stage 1 development ends
                if (SUMDTT > (params['TT_TDU_LIMIT'] * (params['PHINT'] / 95.0)) ):
                    END_VEGETATION_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_VEGETATION_DATE - END_JUVENILE_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    # Sum of the two phases
                    CROP_AGE_2 = str(END_VEGETATION_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE_2) # Sum of the last two phases
                    growstages[f'{ISTAGE}']['date'] = str(END_VEGETATION_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_VEGETATION_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = h_status
                    #if (verbose is True):
                    #    print("End of Juvenile: Thermal time reached at days duration ", i+1,
                    #          str(END_VEGETATION_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                    break
        else:
            print("Error reading weather data for vegetation phase")

        # print("End of Vegeation Growth ", END_VEGETATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of vegetation growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF EAR GROWTH - End of Vegetative Growth to End of Ear Grow (End leaf growth)
    #-----------------------------------------------------------------------------------------------
    ISTAGE = 2 # Terminal spikelet initiation to the end of leaf growth - CERES Stage 2
    try:
        SUMDTT = 0.0
        P2 = params['PHINT'] * 3

        w = weather[weather['DATE']>=END_VEGETATION_DATE].reset_index(drop=True)
        END_OF_EAR_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P2):
                    END_OF_EAR_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_EAR_GROWTH_DATE - END_VEGETATION_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_EAR_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_EAR_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = h_status
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_EAR_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Ear growth",END_OF_EAR_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of ear growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    #
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE ANTHESIS
    # ----------------------------------------------------------------------------------------------
    # Anthesis date was estimated as occurring 7 d after heading. (based on McMaster and Smika, 1988; 
    # McMaster and Wilhelm, 2003; G. S. McMaster, unpubl. data). 
    # Here we used 6 days according to IWIN reported anthesis
    ISTAGE = 2.5
    ADAH = params['ADAH'] # Anthesis days after heading
    
    # Get ADAH parameter
    a_status = 1
    if (params['bruteforce']==True and params['brute_params']!={} and 
        params['brute_params']["obsAnthesisDAP"] is not None):
        # using brute force algorithms
        a_growstages, params, a_status = estimate_anthesis_by_bruteforce(params)
        #params = { **params, **e_params}
    elif ( useDefault is True and params['longitude']!=-180.0 and params['longitude']!='' 
          and params['longitude'] is not None):
        ## using geographic coordinates and month to extract values by linear regression
        params = estimate_anthesis_by_default(config, params, END_OF_EAR_GROWTH_DATE)
    elif (params['longitude']!=-180.0 and params['longitude']!='' and params['longitude'] is not None):
        ## using geographic coordinates
        params = estimate_anthesis_by_coords(config, params)
    elif (params['genotype']!="" and params['genotype'] is not None):
        # using cultivar or genotype name
        params = estimate_anthesis_by_cultivar(config, params, END_OF_EAR_GROWTH_DATE)
    
    CROP_AGE = DAP + ADAH
    ANTHESIS_DATE = END_OF_EAR_GROWTH_DATE + pd.DateOffset(days=ADAH)
    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
    growstages[f'{ISTAGE}']['date'] = str(ANTHESIS_DATE).split(' ')[0]
    growstages[f'{ISTAGE}']['DOY'] = int(ANTHESIS_DATE.dayofyear)
    growstages[f'{ISTAGE}']['SUMDTT'] = -99.0 #round(SUMDTT-100, 1) # TODOs: SUMDTT debe recalcularse
    growstages[f'{ISTAGE}']['DAP'] = int(DAP + ADAH)
    growstages[f'{ISTAGE}']['status'] = a_status
    
    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF PANNICLE GROWTH - End pannicle growth - End of Ear Growth to Start of Grain Filling
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 3 # Preanthesis ear growth - CERES Stage 3.
    try:
        SUMDTT = 0.0 #SUMDTT - P2
        P3 = params['PHINT'] * 2
        #TBASE=0.0

        w = weather[weather['DATE']>END_OF_EAR_GROWTH_DATE].reset_index(drop=True)
        END_OF_PANNICLE_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P3):
                    END_OF_PANNICLE_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_PANNICLE_GROWTH_DATE - END_OF_EAR_GROWTH_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_PANNICLE_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_PANNICLE_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = 1
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_PANNICLE_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Pre-Anthesis Ear growth",END_OF_PANNICLE_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of pre-anthesis earh growth date (end of pannicle growth date).", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE BEGIN GRAIN FILLING - Grain fill - Start of Grain Filling to Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 4 # Preanthesis ear growth to the beginning of grain filling - CERES Stage 4.
    try: 
        P4 = params['DSGFT'] #200 GDD # APSIM-Wheat = 120
        SUMDTT = 0.0 #SUMDTT - P3

        w = weather[weather['DATE']>=END_OF_PANNICLE_GROWTH_DATE].reset_index(drop=True)
        BEGIN_GRAIN_FILLING_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT
                if (SUMDTT >= P4):
                    BEGIN_GRAIN_FILLING_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(BEGIN_GRAIN_FILLING_DATE - END_OF_PANNICLE_GROWTH_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(BEGIN_GRAIN_FILLING_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(BEGIN_GRAIN_FILLING_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = 1
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(BEGIN_GRAIN_FILLING_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("Begining of Grain fill",BEGIN_GRAIN_FILLING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining begin of grain fill date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END GRAIN FILLING - Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 5
    # Get SNOW, VREQ, P1V, P1D,  parameters using geographic coordinates
    m_status = 1
    if (params['bruteforce']==True and params['brute_params']!={} and 
        params['brute_params']["obsMaturityDAP"] is not None):
        # using brute force algorithms
        m_growstages, params, m_status = estimate_maturity_by_bruteforce(params)
    elif ( useDefault is True and params['longitude']!=-180.0 and params['longitude']!='' 
          and params['longitude'] is not None):
        ## using geographic coordinates and month to extract values by linear regression
        params = estimate_maturity_by_default(config, params, BEGIN_GRAIN_FILLING_DATE)
    elif (params['longitude']!=-180.0 and params['longitude']!='' and params['longitude'] is not None):
        ## using geographic coordinates
        params = estimate_maturity_by_coords(config, params)
    elif (params['genotype']!="" and params['genotype'] is not None):
        # using cultivar or genotype name
        params = estimate_maturity_by_cultivar(config, params, BEGIN_GRAIN_FILLING_DATE)
    
    try:
        SUMDTT = 0.0 #SUMDTT - P4
        #P5 = 430 + params['P5'] * 20 # P5 = (0.05 X TT_Maturity) - 21.5. ~500 degree-days
        P5 = params['P5'] # 400 + 5.0 * 20  

        w = weather[weather['DATE']>=BEGIN_GRAIN_FILLING_DATE].reset_index(drop=True)
        END_GRAIN_FILLING_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT
                if (SUMDTT >= P5):
                    END_GRAIN_FILLING_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_GRAIN_FILLING_DATE - BEGIN_GRAIN_FILLING_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_GRAIN_FILLING_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_GRAIN_FILLING_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = m_status
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_GRAIN_FILLING_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Grain filling",END_GRAIN_FILLING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of grain fill date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE HARVEST - Harvest - End of Grain Filling or Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 6  # Physiological maturity to harvest - CERES Stage 6.
    try: 
        SUMDTT = 0.0
        estimateHarvest = True
        P6 = params['P6']

        if (estimateHarvest is False):
            HARVEST = END_GRAIN_FILLING_DATE
            growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
            growstages[f'{ISTAGE}']['date'] = str(HARVEST).split(' ')[0]
            growstages[f'{ISTAGE}']['DOY'] = int(HARVEST.dayofyear)
            growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
            growstages[f'{ISTAGE}']['DAP'] = DAP
        else:
            # A value of 250 degree-days can be used to approximate the thermal time from physiological maturity to harvest
            w = weather[weather['DATE']>=END_GRAIN_FILLING_DATE].reset_index(drop=True)
            HARVEST = ''
            if (len(w)>0):
                for i in range(len(w)):
                    Tmin = float(w.iloc[i]['TMIN'])
                    Tmax = float(w.iloc[i]['TMAX'])
                    # Thermal time
                    DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                                   Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                                   Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                    SUMDTT = SUMDTT + DTT
                    if (SUMDTT >= P6):
                        HARVEST = w.iloc[i]['DATE']
                        CROP_AGE = str(HARVEST - END_GRAIN_FILLING_DATE).replace(' days 00:00:00','')
                        DAP = DAP + int(CROP_AGE)
                        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                        growstages[f'{ISTAGE}']['date'] = str(HARVEST).split(' ')[0]
                        growstages[f'{ISTAGE}']['DOY'] = int(HARVEST.dayofyear)
                        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                        growstages[f'{ISTAGE}']['DAP'] = DAP
                        growstages[f'{ISTAGE}']['status'] = 1
                        #if (verbose is True):
                        #    print("Thermal time reached at days duration ", i+1, str(HARVEST), 
                        #          CROP_AGE, DAP, round(SUMDTT, 1))
                        break
    except Exception as err:
        try:
            raise StageFailed("Problem determining physiological maturity to harvest date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
        
    # ---------------------
    if (dispDates is True):
        try:
            if (params['bruteforce']==True):
                print("RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE       STATUS")
                for i, k in enumerate(growstages.keys()):
                    print("{:4}   {:10} {:>10} {:>4} {:>6} {:>12} {:>12} {:>12}".format(i, growstages[k]['istage_old'], 
                                                                                 growstages[k]['DAP'],
                                                                                 growstages[k]['DOY'], 
                                                                                 growstages[k]['AGE'],
                                                                                 growstages[k]['SUMDTT'], 
                                                                                 growstages[k]['date'],
                                                                                 growstages[k]['status']
                                                                                       ))
            else:
                print("RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE ")
                for i, k in enumerate(growstages.keys()):
                    print("{:4}   {:10} {:>10} {:>4} {:>6} {:>12} {:>12}".format(i, growstages[k]['istage_old'], 
                                                                             growstages[k]['DAP'],
                                                                             growstages[k]['DOY'], growstages[k]['AGE'],
                                                                             growstages[k]['SUMDTT'], 
                                                                             growstages[k]['date']))
        except Exception as err:
            try:
                raise StageFailed("Problem displaying results.", -99, err)
            except StageFailed as x:
                print(x)
                return
            
    if (dispFigPhenology is True):
        try:
            # remove anthesis to display correctly
            if ('2.5' in growstages):
                del growstages['2.5']
            drawPhenology(gs=growstages, title='Phenological growth phases of Wheat', dpi=150,
                         dispPlants=True, topDAPLabel=True, timeSpanLabel=True, topNameStageLabel=True,
                         topNameStageLabelOpt=True, copyrightLabel=True, 
                         saveFig=False, showFig=True, path_to_save_results='./', 
                         fname='Fig_1_Phenological_Phases_Wheat', fmt='jpg')
        except Exception as err:
            try:
                raise StageFailed("Problem displaying figure of phenological stages.", -99, err)
            except StageFailed as x:
                print(x)
                return
    # ---------------------
    return growstages, params

# ==================================

# --------------------- EMERGENCE
''' Estimate emergence using a linear correlation among Thermal time and GDDE '''
def estimate_emergence_by_default(config=None, params=None, GERMINATION_DATE=None):
    if (config is None or params is None or GERMINATION_DATE is None):
        return params
    
    if (params['latitude']==-99.0 or params['longitude']==-180.0):
        return params
    
    s = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d")
    month = int(s.strftime('%m'))
    # Extract values from configuration file "configbycoords"
    #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
    sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
    new_site = pd.DataFrame({"lat": [params['latitude']], 
                             "lon": [params['longitude']],
                             "sowing_date": [sowing_date_ms]
                            }) 
    nearest = config["nn"].kneighbors(new_site, n_neighbors=3, return_distance=False)
    params['SDEPTH'] = float(config["configbycoords"].iloc[nearest[0]]['SDEPTH'].mean())
    emerSUMDTT = float(config["configbycoords"].iloc[nearest[0]]['emerSUMDTT'].mean())
    w2 = params['weather'][( (params['weather']['DATE']>=GERMINATION_DATE) )].reset_index(drop=True)
    if (len(w2)>0):
        # 3. Con el TT utilizamos la función lineal por genotipo y obtenemos GDDE aproximado
        obsTT = 0
        for i in range(len(w2)):
            Tmin = float(w2.iloc[i]['TMIN'])
            Tmax = float(w2.iloc[i]['TMAX'])
            TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            obsTT = obsTT + TT
            
            if (obsTT >= emerSUMDTT):
                break
        #
        if (obsTT < 190):
            # y = 0.1798027x + -5.4839940 # Extract from 1 - Fine-tune Maturity.ipynb
            # y = 0.1496304x + -1.3641063 # < 190. # Sensitivity Analysis Phenology IWIN.ipynb
            #params['GDDE'] = 0.1798027 * obsTT - 5.4839940
            params['GDDE'] = 0.1496304 * obsTT - 1.3641063
        elif ((obsTT > 190) and (obsTT < 400)):
            # y = 0.0989615x + -4.2659372 #> 210 < 400 # Sensitivity Analysis Phenology IWIN.ipynb
            params['GDDE'] = 0.0989615 * obsTT - 4.2659372
        elif (obsTT > 400):
            # y = 0.0655266x + -2.5076657 # > 400 # Sensitivity Analysis Phenology IWIN.ipynb
            params['GDDE'] = 0.0655266 * obsTT - 2.5076657
            
        del w2
        
    del new_site, nearest
    _ = gc.collect()
    
    return params

        
def estimate_emergence_by_coords(config=None, params=None):
    if (config is None or params is None):
        return
    # Extract values from configuration file "configbycoords"
    #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
    sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
    new_site = pd.DataFrame({"lat": [params['latitude']], 
                             "lon": [params['longitude']],
                             #"smonth": [int(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").strftime('%m'))]
                             "sowing_date": [sowing_date_ms]
                            }) # , "emerg_tavg": [9]})
    nearest = config["nn"].kneighbors(new_site, n_neighbors=3, return_distance=False)
    params['GDDE'] = float(config["configbycoords"].iloc[nearest[0]]['GDDE'].mean())
    params['SDEPTH'] = float(config["configbycoords"].iloc[nearest[0]]['SDEPTH'].mean())
    del new_site, nearest
    _ = gc.collect()
    return params

def estimate_emergence_by_cultivar(config=None, params=None, GERMINATION_DATE=None):
    if (config is None or params is None or GERMINATION_DATE is None):
        return
    # Extract values from configuration file "configbygenotype"
    gen_params = config["configbygenotype"][ config["configbygenotype"]['genotype'] == params['genotype']].reset_index(drop=True)
    if (len(gen_params)>0):
        fct1 = gen_params.iloc[0]['emerfct1']
        fct2 = gen_params.iloc[0]['emerfct2']
        MAX_DAP = gen_params.iloc[0]['emerdap']
        params['SDEPTH'] = gen_params.iloc[0]['SDEPTH']
    else:
        print(f"Genotype {params['genotype']} not found. Using generic equation")
        fct1 = 0.15
        fct2 = -1.13
        MAX_DAP = 10
        params['SDEPTH'] = 3.0
    #
    s = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d")
    e = s + pd.DateOffset(days=MAX_DAP) #timedelta(days=MAX_DAP)
    # Estimate the Thermal time to evaluate P9
    w2 = params['weather'][( (params['weather']['DATE']>=GERMINATION_DATE) & (params['weather']['DATE']<=e) )].reset_index(drop=True)
    if (len(w2)>0):
        # 3. Con el TT utilizamos la función lineal por genotipo y obtenemos GDDE aproximado
        obsTT = 0
        for i in range(len(w2)):
            Tmin = float(w2.iloc[i]['TMIN'])
            Tmax = float(w2.iloc[i]['TMAX'])
            TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            obsTT = obsTT + TT
        #
        params['GDDE'] = fct1 * obsTT + fct2
        del w2
    #
    del gen_params
    _ = gc.collect()
    return params

def determine_emergence_stage(initparams=None, verbose=False):
    '''
        Estimate Wheat phenological stages using CERES-Wheat model

        Parameters:
            initparams (dict): A dictionary with initial parameters
            verbose (bool): Display comments during the processes. Default is False
        
        Attributes:
            weather (object): A table or dataframe with weather data for the site
            TT_TBASE (float): Base temperature for estimate Thermal time. Default 0.0
            TT_TEMPERATURE_OPTIMUM (float): Thermal time optimum temperature. Default 26
            TT_TEMPERATURE_MAXIMUM (float): Thermal time maximum temperature. Default 34
            CIVIL_TWILIGHT (float): Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0
            HI (float): Hardiness Index. Default 0.0 
            SNOW (float): Snow fall. Default 0.0
            SDEPTH (float): Sowing depth in cm. Default 3.0 cm
            GDDE (float): Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.
            DAYS_GERMIMATION_LIMIT (float): Threshold for days to germination. Default 40
            TT_EMERGENCE_LIMIT (float): Threshold for thermal time to emergence. Default 300
            
        Returns:
            growstages (dict): A dictionary with all phenological stages and addtional useful information
            
    '''
    if (initparams is None):
        print("Please check out the input parameters")
        return
    
    # Initialization of variables 
    params = dict(
        weather = None, # Weather data of the site
        sowing_date = "", # Sowing date in YYYY-MM-DD
        latitude = -99.0, # Latitude of the site
        TT_TBASE = 0.0, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = 26, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = 34, # Thermal time maximum temperature
        CIVIL_TWILIGHT = 0.0, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        HI = 0.0, # Hardiness Index
        SNOW = 0, # Snow fall
        SDEPTH = 3.0, # Sowing depth in cm
        GDDE = 6.2, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DAYS_GERMIMATION_LIMIT = 40, # threshold for days to germination
        TT_EMERGENCE_LIMIT = 300, # threshold for thermal time to emergence
    )
    if (initparams is not None):
        params = {**params, **initparams}
        
    if (params['sowing_date'] is None or params['sowing_date']==""):
        print("Sowing date not valid")
        return
    if (params['latitude'] is None or params['latitude']==-99.0):
        print("Latitude of the site not valid")
        return
    if (params['weather'] is None):
        print("Weather data is not available")
        return
    else:
        weather = params['weather']
    
    
    growstages = {
            '7': {'istage_old': 'Sowing', 'istage': 'Fallow', 'desc': 'No crop present to Sowing', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '8': {'istage_old': 'Germinate', 'istage': 'Sowing', 'desc': 'Sowing to Germination', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '9': {'istage_old': 'Emergence', 'istage': 'Germinate', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''}
    }
    
    class StageFailed(Exception):
        def __init__(self, m, istage, err):
            self.message = m
            self.istage = istage
            self.err = err
        def __str__(self):
            return self.message + f" Stage ({self.istage}) - " + f"Error: {self.err}"

    # --------------------------------------------------------------------------
    # DETERMINE SOWING DATE
    # --------------------------------------------------------------------------
    ISTAGE = 7
    try:
        SOWING_DATE = pd.to_datetime(str(params['sowing_date']), format='%Y-%m-%d' )
        DOY = pd.to_datetime(SOWING_DATE).dayofyear

        growstages[f'{ISTAGE}']['date'] = str(SOWING_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(DOY)
        growstages[f'{ISTAGE}']['AGE'] = 0
        growstages[f'{ISTAGE}']['SUMDTT'] = 0
        growstages[f'{ISTAGE}']['DAP'] = 0
        #print("Sowing date:", SOWING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem initializing the determination of phenological stage. Please check your input parameters such as sowing date or latitude of the site", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
        
    # --------------------------------------------------------------------------
    # DETERMINE GERMINATION  DATE
    # --------------------------------------------------------------------------
    ISTAGE = 8
    try:
        SUMDTT = 0.0
        #VF = 0.0
        DAP = 0
        ndays = 1 # Seed germination is a rapid process and is assumed to occur in one day
        w = weather[(weather['DATE']==(SOWING_DATE + pd.DateOffset(days=ndays)) )].reset_index(drop=True)
        GERMINATION_DATE = ''
        Tmin = float(w.iloc[ndays-1]['TMIN'])
        Tmax = float(w.iloc[ndays-1]['TMAX'])
        # Thermal time
        DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                       Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                       Ttop=params['TT_TEMPERATURE_MAXIMUM'])
        SUMDTT = SUMDTT + DTT
        GERMINATION_DATE = w.iloc[ndays-1]['DATE']
        CROP_AGE = str(GERMINATION_DATE - SOWING_DATE).replace(' days 00:00:00','')
        DAP = DAP + int(CROP_AGE)
        growstages[f'{ISTAGE}']['date'] = str(GERMINATION_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(GERMINATION_DATE.dayofyear)
        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
        growstages[f'{ISTAGE}']['DAP'] = DAP

        #print("Germination date:", GERMINATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining germination date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return


    # --------------------------------------------------------------------------
    # DETERMINE SEEDLING EMERGENCE DATE
    # --------------------------------------------------------------------------
    ISTAGE = 9
    try:
        P9 = 40 + params['GDDE'] * params['SDEPTH'] #Default values
        SUMDTT = 0.0
        #print("Growing degree days from germination to emergence (P9): ",P9) 
        # The crop will die if germination has not occurred before a certain period (eg. 40 days)
        w = weather[weather['DATE']>=GERMINATION_DATE].reset_index(drop=True)
        EMERGENCE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            SUMDTT = SUMDTT + DTT

            if (SUMDTT >= P9 or SUMDTT > params['TT_EMERGENCE_LIMIT']):
                EMERGENCE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(EMERGENCE_DATE - GERMINATION_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(EMERGENCE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(EMERGENCE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(EMERGENCE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break

        #print("Emergence date: ", EMERGENCE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining emergence date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    #
    return growstages
# ----------------------------------

# Getting Emergence by brute force algorithm
def estimate_emergence_by_bruteforce(params=None):
    '''Getting Emergence date by brute force algorithm
        
        Parameters:
            params (dict): A dictionary with attributes
            
        Attributes:
            obsEmergenceDAP (int): Observed days after planting to emergence.
            sowingdate (str): Sowing date in YYYY-MM-DD format.
            latitude (float): Latitude of the site in celsius degrees
            weather (object): A table or dataframe with weather data for the site
            max_tries (int): Number of maximum tries to find the best value
            error_lim (float): Threshold to classify the observation as a good or bad
            gdde_steps (float): Step to increase or reduce the GDDE parameters. Default 1.0
            maxGDDE (float): Threshold for the maximum value of GDDE to reach emergence date
            TT_EMERGENCE_LIMIT (int): # threshold for thermal time to emergence. Default is 300 degree-days
            
        Returns:
            params (dict): Update params dictionary with GDDE and SDEPTH values for Emergence date using P9
            
    '''
    if (params is None):
        print("Parameters not valid")
        return
    if (params['brute_params']['obsEmergenceDAP'] is None):
        print("Observed emergence days after planting not defined")
        return
    if (params['sowing_date'] is None):
        print("Sowing date not valid")
        return
    if (params['latitude'] is None):
        print("Latitude of the site not valid")
        return
    if (params['weather'] is None):
        print("Weather data not defined")
        return
    try:
        
        # Setup initial parameters
        sowingdate = params['sowing_date']
        latitude = params['latitude']
        weather = params['weather']
        obsDAP = params['brute_params']["obsEmergenceDAP"]
        max_tries = params['brute_params']['max_tries']
        error_lim = params['brute_params']['error_lim']
        gdde_steps = params['brute_params']['gdde_steps']
        maxGDDE = params['brute_params']['maxGDDE']
        TT_EMERGENCE_LIMIT = params['TT_EMERGENCE_LIMIT']
        SDEPTH = params['SDEPTH']
        GDDE = params['GDDE']
        # Run initial simulation
        growstages = determine_emergence_stage(initparams=params)
        # loop until converge
        status = 0
        t = 0
        simDAP = int(growstages['9']['DAP']) 
        while True:
            if (simDAP < obsDAP):
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

            new_params = dict( SDEPTH=SDEPTH, GDDE=GDDE ) # Updated parameters
            params = {**params, **new_params}
            growstages = determine_emergence_stage(initparams=params)

            try:
                simDAP = int(growstages['9']['DAP']) # Problem with DAP = '' # not found
            except:
                status = -1
                break
            if (simDAP == obsDAP):
                status = 1
                break
            elif (int(growstages['9']['SUMDTT']) > TT_EMERGENCE_LIMIT):
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

    except Exception as err:
        print(f"Problem getting emergence by brute force",err)

    return growstages, params, status
# --------------------- END EMERGENCE

# ==================================
        
# --------------------- HEADING
''' Estimate heading using a linear correlation among Thermal time and PHINT '''
def estimate_heading_by_default(config=None, params=None, EMERGENCE_DATE=None):
    if (config is None or params is None or EMERGENCE_DATE is None):
        return params
    
    if (params['latitude']==-99.0 or params['longitude']==-180.0):
        return params
    
    s = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d")
    month = int(s.strftime('%m'))
    # Extract values from configuration file "configbycoords"
    #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
    sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
    new_site = pd.DataFrame({"lat": [params['latitude']], 
                             "lon": [params['longitude']],
                             "sowing_date": [sowing_date_ms]
                            }) 
    nearest = config["nn"].kneighbors(new_site, n_neighbors=3, return_distance=False)
    params['SNOW'] = float(config["configbycoords"].iloc[nearest[0]]['SNOW'].mean())
    params['VREQ'] = float(config["configbycoords"].iloc[nearest[0]]['VREQ'].mean())
    params['P1V'] = float(config["configbycoords"].iloc[nearest[0]]['P1V'].mean())
    params['P1D'] = float(config["configbycoords"].iloc[nearest[0]]['P1D'].mean())
    headSUMDTT = float(config["configbycoords"].iloc[nearest[0]]['headSUMDTT'].mean())
    w2 = params['weather'][( (params['weather']['DATE']>=EMERGENCE_DATE) )].reset_index(drop=True)
    if (len(w2)>0):
        # 3. Con el TT utilizamos la función lineal por genotipo y obtenemos GDDE aproximado
        obsTT = 0
        for i in range(len(w2)):
            Tmin = float(w2.iloc[i]['TMIN'])
            Tmax = float(w2.iloc[i]['TMAX'])
            TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            obsTT = obsTT + TT
            
            if (obsTT >= headSUMDTT): # TODO: Check out if we need to substract P9
                break
        # y = 0.3213695x + 0.7459237 # Extract from 1 - Fine-tune Maturity.ipynb
        # params['PHINT'] = 0.3213695 * obsTT + 0.7459237
        # y = 0.3234813x + 0.2346515 # Sensitivity Analysis Phenology IWIN.ipynb
        params['PHINT'] = 0.3234813 * obsTT + 0.2346515
        del w2
        
    del new_site, nearest
    _ = gc.collect()
    
    return params

def estimate_heading_by_coords(config=None, params=None):
    if (config is None or params is None):
        return
    # Extract values from configuration file "configbycoords"
    if (params['longitude']!=-180.0 and params['longitude']!='' and params['longitude'] is not None):
        # Extract values from configuration file "configbycoords"
        #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
        sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
        new_site = pd.DataFrame({"lat": [params['latitude']], 
                                 "lon": [params['longitude']],
                                 #"smonth": [int(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").strftime('%m'))]
                                 "sowing_date": [sowing_date_ms]
                                }) # , "emerg_tavg": [9]})
        n_neighbors=1
        nearest = config['nn'].kneighbors(new_site, n_neighbors=n_neighbors, return_distance=False)
        if (n_neighbors>1):
            params['SNOW'] = float(config["configbycoords"].iloc[nearest[0]]['SNOW'].mean())
            params['VREQ'] = float(config["configbycoords"].iloc[nearest[0]]['VREQ'].mean())
            params['P1V'] = float(config["configbycoords"].iloc[nearest[0]]['P1V'].mean())
            params['P1D'] = float(config["configbycoords"].iloc[nearest[0]]['P1D'].mean())
            params['PHINT'] = float(config["configbycoords"].iloc[nearest[0]]['PHINT'].mean())
        else:
            params['SNOW'] = float(config["configbycoords"].iloc[nearest[0]]['SNOW'])
            params['VREQ'] = float(config["configbycoords"].iloc[nearest[0]]['VREQ'])
            params['P1V'] = float(config["configbycoords"].iloc[nearest[0]]['P1V'])
            params['P1D'] = float(config["configbycoords"].iloc[nearest[0]]['P1D'])
            params['PHINT'] = float(config["configbycoords"].iloc[nearest[0]]['PHINT'])

    del new_site, nearest
    _ = gc.collect()
    return params

def estimate_heading_by_cultivar(config=None, params=None, EMERGENCE_DATE=None):
    if (config is None or params is None or EMERGENCE_DATE is None):
        return
    # Extract values from configuration file "configbygenotype"
    gen_params = config["configbygenotype"][config["configbygenotype"]['genotype']==params['genotype']].reset_index(drop=True)
    if (len(gen_params)>0):
        headfct1 = gen_params.iloc[0]['headfct1']
        headfct2 = gen_params.iloc[0]['headfct2']
        MAX_DAP = gen_params.iloc[0]['headdap']
        params['SNOW'] = gen_params.iloc[0]['SNOW']
        params['VREQ'] = gen_params.iloc[0]['VREQ']
        params['P1V'] = gen_params.iloc[0]['P1V']
        params['P1D'] = gen_params.iloc[0]['P1D']
        params['PHINT'] = gen_params.iloc[0]['PHINT']
    else:
        print(f"Genotype {params['genotype']} not found. Using generic equation")
        headfct1 = 0.321
        headfct2 = 0.553
        MAX_DAP = 220
        params['SNOW'] = 0.0
        params['VREQ'] = 505.0
        params['P1V'] = 1.0
        params['P1D'] = 3.675
        #params['PHINT'] = 95.0
        #
        #s = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d")
        h = EMERGENCE_DATE + pd.DateOffset(days=MAX_DAP) #timedelta(days=MAX_DAP)
        # Estimate the Thermal time to evaluate P9
        w2 = params['weather'][( (params['weather']['DATE']>=EMERGENCE_DATE) & (params['weather']['DATE']<=h) )].reset_index(drop=True)
        if (len(w2)>0):
            # 3. Con el TT utilizamos la función lineal por genotipo y obtenemos GDDE aproximado
            obsTT = 0
            for i in range(len(w2)):
                Tmin = float(w2.iloc[i]['TMIN'])
                Tmax = float(w2.iloc[i]['TMAX'])
                TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                   Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                   Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                obsTT = obsTT + TT
            #
            #print(f"Thermal time to Emergence: {obsTT}")
            params['PHINT'] = headfct1 * obsTT + headfct2
            del w2
    #
    del gen_params
    _ = gc.collect()
    return params
#
#
# Getting Heading by brute force algorithm
def estimate_heading_by_bruteforce(params=None):
    '''Getting Heading date by brute force algorithm.
    
    This tool allows users to run the model for several parameter values to test, 
    and then identifies which combination of parameter values results in simulated values that show 
    the least deviations from observed values of a specified trait using root-mean-square error (RMSE) as 
    the main criterion.
    
        Parameters:
            params (dict): A dictionary with attributes
            
        Attributes
            obsHeadingDAP (int): Observed days after planting to Heading.
            sowingdate (str): Sowing date in YYYY-MM-DD format.
            latitude (float): Latitude of the site in celsius degrees
            weather (object): A table or dataframe with weather data for the site
            max_tries (int): Number of maximum tries to find the best value
            error_lim (float): Threshold to classify the observation as a good or bad
            phint_steps (float): Step to increase or reduce the PHINT parameters. Default 1.0
            maxPHINT (float): Threshold for the maximum value of PHINT to reach heading date. Default value taken from DSSAT WHCER048.CUL
            TT_TDU_LIMIT (float): Threshold for thermal development units (TDU). Default 400  degree-days
            
        Returns:
            params (dict): Update params dictionary with with SNOW, VREQ, P1V, P1D and PHINT values for Heading date
            
    '''
    if (params is None):
        print("Parameters not valid")
        return
    if (params['brute_params']['obsHeadingDAP'] is None):
        print("Observed heading days after planting not defined")
        return
    if (params['sowing_date'] is None):
        print("Sowing date not valid")
        return
    if (params['latitude'] is None):
        print("Latitude of the site not valid")
        return
    if (params['weather'] is None):
        print("Weather data not defined")
        return
    try:
        # Setup initial parameters
        sowingdate = params['sowing_date']
        latitude = params['latitude']
        weather = params['weather']
        obsDAP = params['brute_params']['obsHeadingDAP']
        max_tries = params['brute_params']['max_tries']
        error_lim = params['brute_params']['error_lim']
        phint_steps = params['brute_params']['phint_steps']
        maxPHINT = params['brute_params']['maxPHINT']
        TT_TDU_LIMIT = params['TT_TDU_LIMIT']
        SDEPTH = params['SDEPTH']
        GDDE = params['GDDE']
        SNOW = params['SNOW'] #0 
        VREQ = params['VREQ'] #505.0 
        PHINT = params['PHINT'] #95.0
        P1V = params['P1V'] #1.0 # Spring wheat
        P1D = params['P1D'] #3.675
        
        growstages = determine_heading_stage(initparams=params)

        # loop until converge
        status = 0
        t = 0
        simDAP = int(growstages['2']['DAP']) 
        while True:
            if (simDAP < obsDAP):
                PHINT = PHINT + phint_steps
                if PHINT > 120.0:
                    P1V = P1V + 0.25
                    P1V = min(P1V, 5.0)
            else:
                PHINT = PHINT - phint_steps
                PHINT = max(PHINT, 0.0)
            #
            new_params = dict( 
                PHINT=PHINT, P1V=P1V, P1D=P1D, VREQ=VREQ, SNOW=SNOW, #TT_TDU_LIMIT = TT_TDU_LIMIT
            )
            params = {**params, **new_params}
            # Run simulation
            growstages = determine_heading_stage(initparams=params)
            #
            try:
                simDAP = int(growstages['2']['DAP']) # Problem with DAP = '' # not found
            except:
                status = -2
                break
            if (simDAP == obsDAP):
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

    except Exception as err:
        print(f"Problem getting heading by brute force",err)

    return growstages, params, status

def determine_heading_stage(initparams=None, verbose=False):
    '''
        Estimate Wheat phenological stages using an improved PyWheat model calibrated 
        with IWIN datasets (ESWYT, IDYN, HTWYT and SAWYT nurseries)

        Parameters:
            initparams (dict): A dictionary with initial parameters
            verbose (bool): Display comments during the processes. Default is False
        
        Attributes:
            weather (object): A table or dataframe with weather data for the site
            TT_TBASE (float): Base temperature for estimate Thermal time. Default 0.0
            TT_TEMPERATURE_OPTIMUM (float): Thermal time optimum temperature. Default 26
            TT_TEMPERATURE_MAXIMUM (float): Thermal time maximum temperature. Default 34
            CIVIL_TWILIGHT (float): Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0
            HI (float): Hardiness Index. Default 0.0 
            SNOW (float): Snow fall. Default 0.0
            SDEPTH (float): Sowing depth in cm. Default 3.0 cm
            GDDE (float): Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.
            DSGFT (float): GDD from End Ear Growth to Start Grain Filling period. Default 200 degree-days
            VREQ  (float): Vernalization required for max.development rate (VDays). Default 505 degree-days
            PHINT (float): Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. Default 95.0 degree-days
            P1V (float): Development genetic coefficients, vernalization. 1 for spring type, 5 for winter type. Default 4.85
            P1D (float): Development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length). Default 3.675
            P5 (float): Grain filling degree days. Old value was divided by 10. Default 500 degree-days.
            P6 (float): Approximate the thermal time from physiological maturity to harvest. Default 250.
            DAYS_GERMIMATION_LIMIT (float): Threshold for days to germination. Default 40
            TT_EMERGENCE_LIMIT (int): Threshold for thermal time to emergence. Default 300 degree-days
            TT_TDU_LIMIT (float): Threshold for thermal development units (TDU). Default 400  degree-days
            ADAH (int): Number of days after heading. A threshold used for anthesis date after planting. Default is 6 days after heading.
            
        Returns:
            growstages (dict): A dictionary with all phenological stages and addtional useful information
            
    '''
    if (initparams is None):
        print("Please check out the input parameters")
        return
    
    # Initialization of variables 
    params = dict(
        weather = None, # Weather data of the site
        sowing_date = "", # Sowing date in YYYY-MM-DD
        latitude = -90.0, # Latitude of the site
        longitude = -180.0, # Longitude of the site
        genotype = "", # Name of the grand parent in IWIN pedigrees database 
        TT_TBASE = 0.0, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = 26, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = 34, # Thermal time maximum temperature
        CIVIL_TWILIGHT = 0.0, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        HI = 0.0, # Hardiness Index
        SNOW = 0, # Snow fall
        SDEPTH = 3.0, # Sowing depth in cm
        GDDE = 6.2, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = 200, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = 505.0, # Vernalization required for max.development rate (VDays)
        PHINT = 95.0, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = 1.0, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = 3.675, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = 500, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = 250, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = 40, # threshold for days to germination
        TT_EMERGENCE_LIMIT = 300, # threshold for thermal time to emergence
        TT_TDU_LIMIT = 400, # threshold for thermal development units (TDU)
        ADAH = 6, # threshold for anthesis date after planting. This is a 6 days after heading.
    )
    if (initparams is not None):
        params = {**params, **initparams}
    
    # Validate
    if (params['sowing_date']=="" or params['sowing_date'] is None):
        print("Sowing date not defined")
        return
    if (params['latitude']==-90.0 or params['latitude'] is None):
        print("Problem with location of the site. Check the geographic coordinates.")
        return
    if (params['weather'] is None):
        print("Weather data is not available")
        return
    else:
        weather = params['weather']
    
    # ---------------------
    # GDD limits
    # ---------------------
    #P2 = params['PHINT'] * 3
    #P3 = params['PHINT'] * 2
    #P4 = params['DSGFT'] #200 # APSIM-Wheat = 120 # GDD from End Ear Growth to Start Grain Filling period
    #P5 = params['P5'] #430 + params['P5'] * 20
    #P6 = params['P5'] #250
    #P9 = 40 + params['GDDE'] * params['SDEPTH'] 
    
    growstages = {
            '7': {'istage_old': 'Sowing', 'istage': 'Fallow', 'desc': 'No crop present to Sowing', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '8': {'istage_old': 'Germinate', 'istage': 'Sowing', 'desc': 'Sowing to Germination', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '9': {'istage_old': 'Emergence', 'istage': 'Germinate', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '1': {'istage_old': 'Term Spklt', 'istage': 'Emergence', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '2': {'istage_old': 'End Veg', 'istage': 'End Juveni', 'desc': 'End of Juvenile to End of Vegetative growth', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''}
    }
    
    class StageFailed(Exception):
        def __init__(self, m, istage, err):
            self.message = m
            self.istage = istage
            self.err = err
        def __str__(self):
            return self.message + f" Stage ({self.istage}) - " + f"Error: {self.err}"

    # --------------------------------------------------------------------------
    # DETERMINE SOWING DATE
    # --------------------------------------------------------------------------
    ISTAGE = 7
    try:
        SOWING_DATE = pd.to_datetime(str(params['sowing_date']), format='%Y-%m-%d' )
        DOY = pd.to_datetime(SOWING_DATE).dayofyear

        growstages[f'{ISTAGE}']['date'] = str(SOWING_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(DOY)
        growstages[f'{ISTAGE}']['AGE'] = 0
        growstages[f'{ISTAGE}']['SUMDTT'] = 0
        growstages[f'{ISTAGE}']['DAP'] = 0
        #print("Sowing date:", SOWING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem initializing the determination of phenological stage. Please check your input parameters such as sowing date or latitude of the site", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
        

    # --------------------------------------------------------------------------
    # DETERMINE GERMINATION  DATE
    # --------------------------------------------------------------------------
    ISTAGE = 8
    try:
        SUMDTT = 0.0
        #VF = 0.0
        DAP = 0
        ndays = 1 # Seed germination is a rapid process and is assumed to occur in one day
        w = weather[(weather['DATE']==(SOWING_DATE + pd.DateOffset(days=ndays)) )].reset_index(drop=True)
        GERMINATION_DATE = ''
        Tmin = float(w.iloc[ndays-1]['TMIN'])
        Tmax = float(w.iloc[ndays-1]['TMAX'])
        # Thermal time
        DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                       Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                       Ttop=params['TT_TEMPERATURE_MAXIMUM'])
        SUMDTT = SUMDTT + DTT
        GERMINATION_DATE = w.iloc[ndays-1]['DATE']
        CROP_AGE = str(GERMINATION_DATE - SOWING_DATE).replace(' days 00:00:00','')
        DAP = DAP + int(CROP_AGE)
        growstages[f'{ISTAGE}']['date'] = str(GERMINATION_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(GERMINATION_DATE.dayofyear)
        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
        growstages[f'{ISTAGE}']['DAP'] = DAP

        #print("Germination date:", GERMINATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining germination date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return


    # --------------------------------------------------------------------------
    # DETERMINE SEEDLING EMERGENCE DATE
    # --------------------------------------------------------------------------
    ISTAGE = 9
    P9 = 40 + params['GDDE'] * params['SDEPTH']
    try:
        SUMDTT = 0.0
        #print("Growing degree days from germination to emergence (P9): ",P9) 
        # The crop will die if germination has not occurred before a certain period (eg. 40 days)
        
        EMERGENCE_DATE = ''
        w = weather[weather['DATE']>=GERMINATION_DATE].reset_index(drop=True)
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            SUMDTT = SUMDTT + DTT

            if (SUMDTT >= P9 or SUMDTT > params['TT_EMERGENCE_LIMIT']):
                EMERGENCE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(EMERGENCE_DATE - GERMINATION_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(EMERGENCE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(EMERGENCE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(EMERGENCE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break

        #print("Emergence date: ", EMERGENCE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining emergence date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------------------
    # DETERMINE DURATION OF VEGETATIVE PHASE (END JUVENILE DATE - END OF VEGETATION GROWTH
    # --------------------------------------------------------------------------------------
    ISTAGE = 1
    try: 
        isVernalization = True
        SUMDTT = SUMDTT - P9 
        CUMVD = 0
        TDU = 0
        DF = 0.001
        
        w = weather[weather['DATE']>=EMERGENCE_DATE].reset_index(drop=True)
        END_JUVENILE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            if (isVernalization is True):
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                if (CUMVD < params['VREQ']):
                    VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                    if (VF < 0.3):
                        TDU = TDU + DTT * min(VF, DF)
                    else:
                        DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                        TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                        DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                        TDU = TDU + DTT * min(VF, DF)
                    SUMDTT = TDU
                else:
                    isVernalization = False
            else:
                SUMDTT = SUMDTT + DTT

            if (SUMDTT > P9 ): #or SUMDTT > TT_emergence when reached the lower TT
                END_JUVENILE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(END_JUVENILE_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(END_JUVENILE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(END_JUVENILE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(END_JUVENILE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break
            #if (DTT > params['TT_EMERGENCE_LIMIT']): # TT_EMERGENCE_LIMIT = 300,
            #    # The crop will die if germination has not occurred before a certain period (eg. 40 days or 300oC d)
            #    print("The crop died because emergence has not occurred before {} degree-days".format(params['TT_EMERGENCE_LIMIT']))

        #print("End Juvenile date: ", END_JUVENILE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of juvenile date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------
    # DETERMINE END VEGETATION DATE - End of Juvenile to End of Vegetative growth
    # --------------------------------------------------------------------------
    ISTAGE = 1 # <- Note: this must continue with 1 as previous stage (Term Spklt = Emergence to End of Juvenile + End of Juvenile to End of Vegetative growth)
    try:
        isVernalization = True
        VF = 1.0
        w = weather[weather['DATE']>=END_JUVENILE_DATE].reset_index(drop=True)
        END_VEGETATION_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                if (isVernalization is True):
                    Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                    CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                    if (CUMVD < params['VREQ']):
                        VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                        if (VF < 0.3):
                            TDU = TDU + DTT * min(VF, DF)
                        else:
                            DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                            TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                            DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                            TDU = TDU + DTT * min(VF, DF)
                        SUMDTT = TDU
                    else:
                        isVernalization = False
                else:
                    SUMDTT = SUMDTT + DTT

                # When this reduced thermal time accumulation (TDU) reaches 
                # 400 degree days, Stage 1 development ends
                if (SUMDTT > (params['TT_TDU_LIMIT'] * (params['PHINT'] / 95.0)) ):
                    END_VEGETATION_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_VEGETATION_DATE - END_JUVENILE_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    # Sum of the two phases
                    CROP_AGE_2 = str(END_VEGETATION_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE_2) # Sum of the last two phases
                    growstages[f'{ISTAGE}']['date'] = str(END_VEGETATION_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_VEGETATION_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("End of Juvenile: Thermal time reached at days duration ", i+1,
                    #          str(END_VEGETATION_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                    break
        else:
            print("Error reading weather data for vegetation phase")

        # print("End of Vegeation Growth ", END_VEGETATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of vegetation growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF EAR GROWTH - End of Vegetative Growth to End of Ear Grow (End leaf growth)
    #-----------------------------------------------------------------------------------------------
    ISTAGE = 2 # Terminal spikelet initiation to the end of leaf growth - CERES Stage 2
    try:
        SUMDTT = 0.0
        P2 = params['PHINT'] * 3

        w = weather[weather['DATE']>=END_VEGETATION_DATE].reset_index(drop=True)
        END_OF_EAR_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P2):
                    END_OF_EAR_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_EAR_GROWTH_DATE - END_VEGETATION_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_EAR_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_EAR_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_EAR_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Ear growth",END_OF_EAR_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of ear growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    #
    return growstages
# -------------------- END HEADING

# --------------------- ANTHESIS
''' Estimate anthesis using a linear correlation among Thermal time and ADAH 
    Note: This is not necessary due to ADAH is a constant usually of 6 or 10.
'''
def estimate_anthesis_by_default(config=None, params=None, END_OF_EAR_GROWTH_DATE=None):
    if (config is None or params is None or END_OF_EAR_GROWTH_DATE is None):
        return params
    
    if (params['latitude']==-99.0 or params['longitude']==-180.0):
        return params
    
    s = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d")
    month = int(s.strftime('%m'))
    # Extract values from configuration file "configbycoords"
    #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
    sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
    new_site = pd.DataFrame({"lat": [params['latitude']], 
                             "lon": [params['longitude']],
                             "sowing_date": [sowing_date_ms]
                            }) 
    nearest = config["nn"].kneighbors(new_site, n_neighbors=3, return_distance=False)
    anthesisSUMDTT = float(config["configbycoords"].iloc[nearest[0]]['anthesisSUMDTT'].mean())
    w2 = params['weather'][( (params['weather']['DATE']>=END_OF_EAR_GROWTH_DATE) )].reset_index(drop=True)
    if (len(w2)>0):
        obsTT = 0
        for i in range(len(w2)):
            Tmin = float(w2.iloc[i]['TMIN'])
            Tmax = float(w2.iloc[i]['TMAX'])
            TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            obsTT = obsTT + TT
            
            if (obsTT >= anthesisSUMDTT):
                break
        # y = 0.0000170x + 6.0014906 # Extract from 1 - Fine-tune Maturity.ipynb
        # params['ADAH'] = round((0.0000170 * obsTT + 6.0014906), 0)
        # Sensitivity Analysis Phenology IWIN.ipynb
        # ADAH = -0.0312374 x obsTT + 15.2960054
        # PHINT = 0.2984263 x obsTT + 9.0077989
        params['ADAH'] = 5.0 #round((0.0000170 * obsTT + 6.0014906), 0)
        del w2
        
    del new_site, nearest
    _ = gc.collect()
    
    return params

def estimate_anthesis_by_coords(config=None, params=None):
    if (config is None or params is None):
        return
    # Extract values from configuration file "configbycoords"
    if (params['longitude']!=-180.0 and params['longitude']!='' and params['longitude'] is not None):
        # Extract values from configuration file "configbycoords"
        #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
        sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
        new_site = pd.DataFrame({"lat": [params['latitude']], 
                                 "lon": [params['longitude']],
                                 #"smonth": [int(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").strftime('%m'))]
                                 "sowing_date": [sowing_date_ms]
                                }) # , "emerg_tavg": [9]})
        n_neighbors=1
        nearest = config['nn'].kneighbors(new_site, n_neighbors=n_neighbors, return_distance=False)
        # 'lat', 'lon', 'headdap', 'snow', 'vreq', 'PHINT', 'P1V', 'P1D',
        # 'headsumdtt', 'emerdap', 'gdde', 'SDEPTH', 'emersumdtt'
        if (n_neighbors>1):
            params['ADAH'] = float(config["configbycoords"].iloc[nearest[0]]['ADAH'].mean())
        else:
            params['ADAH'] = float(config["configbycoords"].iloc[nearest[0]]['ADAH'])

    del new_site, nearest
    _ = gc.collect()
    return params

def estimate_anthesis_by_cultivar(config=None, params=None, END_OF_EAR_GROWTH_DATE=None):
    if (config is None or params is None or END_OF_EAR_GROWTH_DATE is None):
        return
    # Extract values from configuration file "configbygenotype"
    gen_params = config["configbygenotype"][config["configbygenotype"]['genotype']==params['genotype']].reset_index(drop=True)
    if (len(gen_params)>0):
        headfct1 = gen_params.iloc[0]['headfct1']
        headfct2 = gen_params.iloc[0]['headfct2']
        MAX_DAP = gen_params.iloc[0]['headdap']
        params['ADAH'] = gen_params.iloc[0]['ADAH']
    else:
        print(f"Genotype {params['genotype']} not found. Using generic equation")
        headfct1 = 0.321
        headfct2 = 0.553
        MAX_DAP = 10
        #params['ADAH'] = 10.0
        #
        a = END_OF_EAR_GROWTH_DATE + pd.DateOffset(days=MAX_DAP)  
        w2 = params['weather'][( (params['weather']['DATE']>=END_OF_EAR_GROWTH_DATE) & (params['weather']['DATE']<=a) )].reset_index(drop=True)
        if (len(w2)>0):
            # 3. Con el TT utilizamos la función lineal por genotipo y obtenemos GDDE aproximado
            obsTT = 0
            for i in range(len(w2)):
                Tmin = float(w2.iloc[i]['TMIN'])
                Tmax = float(w2.iloc[i]['TMAX'])
                TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                   Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                   Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                obsTT = obsTT + TT
            #
            params['ADAH'] = headfct1 * obsTT + headfct2
            del w2
    #
    del gen_params
    _ = gc.collect()
    return params
#
#
# Getting anthesis by brute force algorithm
def estimate_anthesis_by_bruteforce(params=None):
    '''Getting Anthesis date by brute force algorithm
    
        Parameters:
            params (dict): A dictionary with attributes
            
        Attributes
            obsAnthesisDAP (int): Observed days after planting to Anthesis.
            sowingdate (str): Sowing date in YYYY-MM-DD format.
            latitude (float): Latitude of the site in celsius degrees
            weather (object): A table or dataframe with weather data for the site
            max_tries (int): Number of maximum tries to find the best value
            error_lim (float): Threshold to classify the observation as a good or bad
            adap_steps (int): Step to increase or reduce the ADAH parameters. Default 1
            maxADAP (float): Threshold for the maximum value of ADAH to reach anthesis date.
            ADAH (int): Number of days after heading. A threshold used for anthesis date after planting. Default is 6 days after heading.
            
        Returns:
            params (dict): Update params dictionary with with ADAH value for Anthesis date
            
    '''
    if (params is None):
        print("Parameters not valid")
        return
    if (params['brute_params']['obsAnthesisDAP'] is None):
        print("Observed anthesis days after planting not defined")
        return
    if (params['sowing_date'] is None):
        print("Sowing date not valid")
        return
    if (params['latitude'] is None):
        print("Latitude of the site not valid")
        return
    if (params['weather'] is None):
        print("Weather data not defined")
        return
    try:
        # Setup initial parameters
        sowingdate = params['sowing_date']
        latitude = params['latitude']
        weather = params['weather']
        obsDAP = params['brute_params']['obsAnthesisDAP']
        max_tries = params['brute_params']['max_tries']
        error_lim = params['brute_params']['error_lim']
        adap_steps = params['brute_params']['adap_steps']
        maxADAP = params['brute_params']['maxADAP']
        ADAH = params['ADAH']
        
        growstages = determine_anthesis_stage(initparams=params)

        # loop until converge
        status = 0
        t = 0
        simDAP = int(growstages['2.5']['DAP']) 
        while True:
            if (simDAP < obsDAP):
                ADAH = ADAH + adap_steps
                #ADAH = min(maxADAP, ADAH)
            else:
                ADAH = ADAH - adap_steps
                ADAH = max(ADAH, 0.0)
            #
            new_params = dict( 
                ADAH=ADAH
            )
            params = {**params, **new_params}
            # Run simulation
            growstages = determine_anthesis_stage(initparams=params)
            #
            try:
                simDAP = int(growstages['2.5']['DAP']) # Problem with DAP = '' # not found
            except:
                status = -3
                break
            if (simDAP == obsDAP):
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

    except Exception as err:
        print(f"Problem getting anthesis",err)

    return growstages, params, status


def determine_anthesis_stage(initparams=None, verbose=False):
    '''
        Estimate Wheat phenological stages using an improved PyWheat model calibrated 
        with IWIN datasets (ESWYT, IDYN, HTWYT and SAWYT nurseries)

        Parameters:
            initparams (dict): A dictionary with initial parameters
            verbose (bool): Display comments during the processes. Default is False
        
        Attributes:
            weather (object): A table or dataframe with weather data for the site
            TT_TBASE (float): Base temperature for estimate Thermal time. Default 0.0
            TT_TEMPERATURE_OPTIMUM (float): Thermal time optimum temperature. Default 26
            TT_TEMPERATURE_MAXIMUM (float): Thermal time maximum temperature. Default 34
            CIVIL_TWILIGHT (float): Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0
            HI (float): Hardiness Index. Default 0.0 
            SNOW (float): Snow fall. Default 0.0
            SDEPTH (float): Sowing depth in cm. Default 3.0 cm
            GDDE (float): Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.
            DSGFT (float): GDD from End Ear Growth to Start Grain Filling period. Default 200 degree-days
            VREQ  (float): Vernalization required for max.development rate (VDays). Default 505 degree-days
            PHINT (float): Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. Default 95.0 degree-days
            P1V (float): Development genetic coefficients, vernalization. 1 for spring type, 5 for winter type. Default 4.85
            P1D (float): Development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length). Default 3.675
            P5 (float): Grain filling degree days. Old value was divided by 10. Default 500 degree-days.
            P6 (float): Approximate the thermal time from physiological maturity to harvest. Default 250.
            DAYS_GERMIMATION_LIMIT (float): Threshold for days to germination. Default 40
            TT_EMERGENCE_LIMIT (int): Threshold for thermal time to emergence. Default 300 degree-days
            TT_TDU_LIMIT (float): Threshold for thermal development units (TDU). Default 400  degree-days
            ADAH (int): Number of days after heading. A threshold used for anthesis date after planting. Default is 6 days after heading.
            
        Returns:
            growstages (dict): A dictionary with all phenological stages and addtional useful information
            
    '''
    if (initparams is None):
        print("Please check out the input parameters")
        return
    
    # Initialization of variables 
    params = dict(
        weather = None, # Weather data of the site
        sowing_date = "", # Sowing date in YYYY-MM-DD
        latitude = -90.0, # Latitude of the site
        longitude = -180.0, # Longitude of the site
        genotype = "", # Name of the grand parent in IWIN pedigrees database 
        TT_TBASE = 0.0, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = 26, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = 34, # Thermal time maximum temperature
        CIVIL_TWILIGHT = 0.0, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        HI = 0.0, # Hardiness Index
        SNOW = 0, # Snow fall
        SDEPTH = 3.0, # Sowing depth in cm
        GDDE = 6.2, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = 200, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = 505.0, # Vernalization required for max.development rate (VDays)
        PHINT = 95.0, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = 1.0, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = 3.675, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = 500, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = 250, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = 40, # threshold for days to germination
        TT_EMERGENCE_LIMIT = 300, # threshold for thermal time to emergence
        TT_TDU_LIMIT = 400, # threshold for thermal development units (TDU)
        ADAH = 6, # threshold for anthesis date after planting. This is a 6 days after heading.
    )
    if (initparams is not None):
        params = {**params, **initparams}
    
    # Validate
    if (params['sowing_date']=="" or params['sowing_date'] is None):
        print("Sowing date not defined")
        return
    if (params['latitude']==-90.0 or params['latitude'] is None):
        print("Problem with location of the site. Check the geographic coordinates.")
        return
    if (params['weather'] is None):
        print("Weather data is not available")
        return
    else:
        weather = params['weather']
    
    # ---------------------
    # GDD limits
    # ---------------------
    #P3 = params['PHINT'] * 2
    #P4 = params['DSGFT'] #200 # APSIM-Wheat = 120 # GDD from End Ear Growth to Start Grain Filling period
    
    growstages = {
            '7': {'istage_old': 'Sowing', 'istage': 'Fallow', 'desc': 'No crop present to Sowing', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '8': {'istage_old': 'Germinate', 'istage': 'Sowing', 'desc': 'Sowing to Germination', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '9': {'istage_old': 'Emergence', 'istage': 'Germinate', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '1': {'istage_old': 'Term Spklt', 'istage': 'Emergence', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '2': {'istage_old': 'End Veg', 'istage': 'End Juveni', 'desc': 'End of Juvenile to End of Vegetative growth', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '2.5': {'istage_old': 'Anthesis', 'istage': 'Anthesis', 'desc': 'Anthesis', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '3': {'istage_old': 'End Ear Gr', 'istage': 'End Veg', 'desc': 'End of Vegetative Growth to End of Ear Grow', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            
    }
    
    class StageFailed(Exception):
        def __init__(self, m, istage, err):
            self.message = m
            self.istage = istage
            self.err = err
        def __str__(self):
            return self.message + f" Stage ({self.istage}) - " + f"Error: {self.err}"

    # --------------------------------------------------------------------------
    # DETERMINE SOWING DATE
    # --------------------------------------------------------------------------
    ISTAGE = 7
    try:
        SOWING_DATE = pd.to_datetime(str(params['sowing_date']), format='%Y-%m-%d' )
        DOY = pd.to_datetime(SOWING_DATE).dayofyear

        growstages[f'{ISTAGE}']['date'] = str(SOWING_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(DOY)
        growstages[f'{ISTAGE}']['AGE'] = 0
        growstages[f'{ISTAGE}']['SUMDTT'] = 0
        growstages[f'{ISTAGE}']['DAP'] = 0
        #print("Sowing date:", SOWING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem initializing the determination of phenological stage. Please check your input parameters such as sowing date or latitude of the site", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
        

    # --------------------------------------------------------------------------
    # DETERMINE GERMINATION  DATE
    # --------------------------------------------------------------------------
    ISTAGE = 8
    try:
        SUMDTT = 0.0
        #VF = 0.0
        DAP = 0
        ndays = 1 # Seed germination is a rapid process and is assumed to occur in one day
        w = weather[(weather['DATE']==(SOWING_DATE + pd.DateOffset(days=ndays)) )].reset_index(drop=True)
        GERMINATION_DATE = ''
        Tmin = float(w.iloc[ndays-1]['TMIN'])
        Tmax = float(w.iloc[ndays-1]['TMAX'])
        # Thermal time
        DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                       Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                       Ttop=params['TT_TEMPERATURE_MAXIMUM'])
        SUMDTT = SUMDTT + DTT
        GERMINATION_DATE = w.iloc[ndays-1]['DATE']
        CROP_AGE = str(GERMINATION_DATE - SOWING_DATE).replace(' days 00:00:00','')
        DAP = DAP + int(CROP_AGE)
        growstages[f'{ISTAGE}']['date'] = str(GERMINATION_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(GERMINATION_DATE.dayofyear)
        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
        growstages[f'{ISTAGE}']['DAP'] = DAP

        #print("Germination date:", GERMINATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining germination date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return


    # --------------------------------------------------------------------------
    # DETERMINE SEEDLING EMERGENCE DATE
    # --------------------------------------------------------------------------
    ISTAGE = 9
    P9 = 40 + params['GDDE'] * params['SDEPTH']
    try:
        SUMDTT = 0.0
        #print("Growing degree days from germination to emergence (P9): ",P9) 
        # The crop will die if germination has not occurred before a certain period (eg. 40 days)
        
        EMERGENCE_DATE = ''
        w = weather[weather['DATE']>=GERMINATION_DATE].reset_index(drop=True)
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            SUMDTT = SUMDTT + DTT

            if (SUMDTT >= P9 or SUMDTT > params['TT_EMERGENCE_LIMIT']):
                EMERGENCE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(EMERGENCE_DATE - GERMINATION_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(EMERGENCE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(EMERGENCE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(EMERGENCE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break

        #print("Emergence date: ", EMERGENCE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining emergence date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------------------
    # DETERMINE DURATION OF VEGETATIVE PHASE (END JUVENILE DATE - END OF VEGETATION GROWTH
    # --------------------------------------------------------------------------------------
    ISTAGE = 1
    try: 
        isVernalization = True
        SUMDTT = SUMDTT - P9 
        CUMVD = 0
        TDU = 0
        DF = 0.001
        
        w = weather[weather['DATE']>=EMERGENCE_DATE].reset_index(drop=True)
        END_JUVENILE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            if (isVernalization is True):
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                if (CUMVD < params['VREQ']):
                    VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                    if (VF < 0.3):
                        TDU = TDU + DTT * min(VF, DF)
                    else:
                        DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                        TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                        DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                        TDU = TDU + DTT * min(VF, DF)
                    SUMDTT = TDU
                else:
                    isVernalization = False
            else:
                SUMDTT = SUMDTT + DTT

            if (SUMDTT > P9 ): #or SUMDTT > TT_emergence when reached the lower TT
                END_JUVENILE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(END_JUVENILE_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(END_JUVENILE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(END_JUVENILE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(END_JUVENILE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break
            #if (DTT > params['TT_EMERGENCE_LIMIT']): # TT_EMERGENCE_LIMIT = 300,
            #    # The crop will die if germination has not occurred before a certain period (eg. 40 days or 300oC d)
            #    print("The crop died because emergence has not occurred before {} degree-days".format(params['TT_EMERGENCE_LIMIT']))

        #print("End Juvenile date: ", END_JUVENILE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of juvenile date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------
    # DETERMINE END VEGETATION DATE - End of Juvenile to End of Vegetative growth
    # --------------------------------------------------------------------------
    ISTAGE = 1 # <- Note: this must continue with 1 as previous stage (Term Spklt = Emergence to End of Juvenile + End of Juvenile to End of Vegetative growth)
    try:
        isVernalization = True
        VF = 1.0
        w = weather[weather['DATE']>=END_JUVENILE_DATE].reset_index(drop=True)
        END_VEGETATION_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                if (isVernalization is True):
                    Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                    CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                    if (CUMVD < params['VREQ']):
                        VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                        if (VF < 0.3):
                            TDU = TDU + DTT * min(VF, DF)
                        else:
                            DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                            TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                            DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                            TDU = TDU + DTT * min(VF, DF)
                        SUMDTT = TDU
                    else:
                        isVernalization = False
                else:
                    SUMDTT = SUMDTT + DTT

                # When this reduced thermal time accumulation (TDU) reaches 
                # 400 degree days, Stage 1 development ends
                if (SUMDTT > (params['TT_TDU_LIMIT'] * (params['PHINT'] / 95.0)) ):
                    END_VEGETATION_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_VEGETATION_DATE - END_JUVENILE_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    # Sum of the two phases
                    CROP_AGE_2 = str(END_VEGETATION_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE_2) # Sum of the last two phases
                    growstages[f'{ISTAGE}']['date'] = str(END_VEGETATION_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_VEGETATION_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("End of Juvenile: Thermal time reached at days duration ", i+1,
                    #          str(END_VEGETATION_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                    break
        else:
            print("Error reading weather data for vegetation phase")

        # print("End of Vegeation Growth ", END_VEGETATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of vegetation growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF EAR GROWTH - End of Vegetative Growth to End of Ear Grow (End leaf growth)
    #-----------------------------------------------------------------------------------------------
    ISTAGE = 2 # Terminal spikelet initiation to the end of leaf growth - CERES Stage 2
    try:
        SUMDTT = 0.0
        P2 = params['PHINT'] * 3

        w = weather[weather['DATE']>=END_VEGETATION_DATE].reset_index(drop=True)
        END_OF_EAR_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P2):
                    END_OF_EAR_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_EAR_GROWTH_DATE - END_VEGETATION_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_EAR_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_EAR_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_EAR_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Ear growth",END_OF_EAR_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of ear growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE ANTHESIS
    # ----------------------------------------------------------------------------------------------
    # Anthesis date was estimated as occurring 7 d after heading. (based on McMaster and Smika, 1988; McMaster and Wilhelm, 2003; G. S. McMaster, unpubl. data)
    # Here we used 6 days according to IWIN reported anthesis
    ISTAGE = 2.5
    ADAH = params['ADAH']
    CROP_AGE = DAP + ADAH
    ANTHESIS_DATE = END_OF_EAR_GROWTH_DATE + pd.DateOffset(days=ADAH)
    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
    growstages[f'{ISTAGE}']['date'] = str(ANTHESIS_DATE).split(' ')[0]
    growstages[f'{ISTAGE}']['DOY'] = int(ANTHESIS_DATE.dayofyear)
    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1) # TODOs: Esto debe recalcularse
    growstages[f'{ISTAGE}']['DAP'] = DAP + ADAH
    
    #
    return growstages
# -------------------- END ANTHESIS


# --------------------- MATURITY
''' Estimate maturity using a linear correlation among Thermal time and P5 '''
def estimate_maturity_by_default(config=None, params=None, BEGIN_GRAIN_FILLING_DATE=None):
    if (config is None or params is None or BEGIN_GRAIN_FILLING_DATE is None):
        return params
    
    if (params['latitude']==-99.0 or params['longitude']==-180.0):
        return params
    
    s = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d")
    month = int(s.strftime('%m'))
    # Extract values from configuration file "configbycoords"
    #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
    sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
    new_site = pd.DataFrame({"lat": [params['latitude']], 
                             "lon": [params['longitude']],
                             "sowing_date": [sowing_date_ms]
                            }) 
    nearest = config["nn"].kneighbors(new_site, n_neighbors=3, return_distance=False)
    maturitySUMDTT = float(config["configbycoords"].iloc[nearest[0]]['maturitySUMDTT'].mean())
    w2 = params['weather'][( (params['weather']['DATE']>=BEGIN_GRAIN_FILLING_DATE) )].reset_index(drop=True)
    if (len(w2)>0):
        # 3. Con el TT utilizamos la función lineal por genotipo y obtenemos GDDE aproximado
        obsTT = 0
        for i in range(len(w2)):
            Tmin = float(w2.iloc[i]['TMIN'])
            Tmax = float(w2.iloc[i]['TMAX'])
            TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            obsTT = obsTT + TT
            
            if (obsTT >= maturitySUMDTT):
                break
        # y = 0.9336157x + 23.2347351 # Extract from 1 - Fine-tune Maturity.ipynb
        # params['P5'] = 0.9336157 * obsTT + 23.2347351
        # y = 0.9606581x + 9.2938099 # Sensitivity Analysis Phenology IWIN.ipynb
        params['P5'] = 0.9606581 * obsTT + 9.2938099
        del w2
        
    del new_site, nearest
    _ = gc.collect()
    
    return params

def estimate_maturity_by_coords(config=None, params=None):
    if (config is None or params is None):
        return
    # Extract values from configuration file "configbycoords"
    if (params['longitude']!=-180.0 and params['longitude']!='' and params['longitude'] is not None):
        # Extract values from configuration file "configbycoords"
        #sowing_date_ms = dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").astype(np.int64) / int(1e6)
        sowing_date_ms = time.mktime(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").timetuple())*1000
        new_site = pd.DataFrame({"lat": [params['latitude']], 
                                 "lon": [params['longitude']],
                                 #"smonth": [int(dt.datetime.strptime(params['sowing_date'], "%Y-%m-%d").strftime('%m'))]
                                 "sowing_date": [sowing_date_ms]
                                }) # , "emerg_tavg": [9]})
        n_neighbors=1
        nearest = config['nn'].kneighbors(new_site, n_neighbors=n_neighbors, return_distance=False)
        # 'lat', 'lon', 'headdap', 'snow', 'vreq', 'PHINT', 'P1V', 'P1D',
        # 'headsumdtt', 'emerdap', 'gdde', 'SDEPTH', 'emersumdtt'
        if (n_neighbors>1):
            params['P5'] = float(config["configbycoords"].iloc[nearest[0]]['P5'].mean())
        else:
            params['P5'] = float(config["configbycoords"].iloc[nearest[0]]['P5'])

    del new_site, nearest
    _ = gc.collect()
    return params

def estimate_maturity_by_cultivar(config=None, params=None, BEGIN_GRAIN_FILLING_DATE=None):
    if (config is None or params is None or BEGIN_GRAIN_FILLING_DATE is None):
        return
    # Extract values from configuration file "configbygenotype"
    gen_params = config["configbygenotype"][config["configbygenotype"]['genotype']==params['genotype']].reset_index(drop=True)
    if (len(gen_params)>0):
        headfct1 = gen_params.iloc[0]['headfct1']
        headfct2 = gen_params.iloc[0]['headfct2']
        MAX_DAP = gen_params.iloc[0]['headdap']
        params['P5'] = gen_params.iloc[0]['P5']
    else:
        print(f"Genotype {params['genotype']} not found. Using generic equation")
        headfct1 = 0.321
        headfct2 = 0.553
        MAX_DAP = 10
        #params['P5'] = 10.0
        #
        a = BEGIN_GRAIN_FILLING_DATE + pd.DateOffset(days=MAX_DAP)  
        w2 = params['weather'][( (params['weather']['DATE']>=BEGIN_GRAIN_FILLING_DATE) & (params['weather']['DATE']<=a) )].reset_index(drop=True)
        if (len(w2)>0):
            # 3. Con el TT utilizamos la función lineal por genotipo y obtenemos GDDE aproximado
            obsTT = 0
            for i in range(len(w2)):
                Tmin = float(w2.iloc[i]['TMIN'])
                Tmax = float(w2.iloc[i]['TMAX'])
                TT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                   Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                   Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                obsTT = obsTT + TT
            #
            params['P5'] = headfct1 * obsTT + headfct2
            del w2
    #
    del gen_params
    _ = gc.collect()
    return params
#
#
# Getting maturity by brute force algorithm
def estimate_maturity_by_bruteforce(params=None):
    '''Getting maturity date by brute force algorithm
    
        Parameters:
            params (dict): A dictionary with attributes
            
        Attributes
            obsMaturityDAP (int): Observed days after planting to maturity.
            sowingdate (str): Sowing date in YYYY-MM-DD format.
            latitude (float): Latitude of the site in celsius degrees
            weather (object): A table or dataframe with weather data for the site
            max_tries (int): Number of maximum tries to find the best value
            error_lim (float): Threshold to classify the observation as a good or bad
            p5_steps (float): Step to increase or reduce the P5 parameters. Default 1.0
            maxP5 (float): Threshold for the maximum value of P5 to reach maturity date.
            
        Returns:
            params (dict): Update params dictionary with with P5 value for maturity date
            
    '''
    if (params is None):
        print("Parameters not valid")
        return
    if (params['brute_params']['obsMaturityDAP'] is None):
        print("Observed maturity days after planting not defined")
        return
    if (params['sowing_date'] is None):
        print("Sowing date not valid")
        return
    if (params['latitude'] is None):
        print("Latitude of the site not valid")
        return
    if (params['weather'] is None):
        print("Weather data not defined")
        return
    try:
        # Setup initial parameters
        sowingdate = params['sowing_date']
        latitude = params['latitude']
        weather = params['weather']
        obsDAP = params['brute_params']['obsMaturityDAP']
        max_tries = params['brute_params']['max_tries']
        error_lim = params['brute_params']['error_lim']
        p5_steps = params['brute_params']['p5_steps']
        maxP5 = params['brute_params']['maxP5']
        P4 = params['DSGFT'] #200 GDD # APSIM-Wheat = 120
        P5 = params['P5'] # P5 = (0.05 X TT_Maturity) - 21.5. ~500 degree-days 
        
        growstages = determine_maturity_stage(initparams=params)

        # loop until converge
        status = 0
        t = 0
        simDAP = int(growstages['5']['DAP'])
        while True:
            if (simDAP < obsDAP):
                P5 = P5 + p5_steps
                #P5 = min(maxP5, P5)
            else:
                P5 = P5 - p5_steps
                P5 = max(P5, 0.0)
            #
            new_params = dict( 
                P5=P5
            )
            params = {**params, **new_params}
            # Run simulation
            growstages = determine_maturity_stage(initparams=params)
            #
            try:
                simDAP = int(growstages['5']['DAP']) # Problem with DAP = '' # not found
            except:
                status = -4
                break
            if (simDAP == obsDAP):
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

    except Exception as err:
        print(f"Problem getting maturity",err)

    return growstages, params, status


def determine_maturity_stage(initparams=None, verbose=False):
    '''
        Estimate Wheat phenological stages using an improved PyWheat model calibrated 
        with IWIN datasets (ESWYT, IDYN, HTWYT and SAWYT nurseries)

        Parameters:
            initparams (dict): A dictionary with initial parameters
            verbose (bool): Display comments during the processes. Default is False
        
        Attributes:
            weather (object): A table or dataframe with weather data for the site
            TT_TBASE (float): Base temperature for estimate Thermal time. Default 0.0
            TT_TEMPERATURE_OPTIMUM (float): Thermal time optimum temperature. Default 26
            TT_TEMPERATURE_MAXIMUM (float): Thermal time maximum temperature. Default 34
            CIVIL_TWILIGHT (float): Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0
            HI (float): Hardiness Index. Default 0.0 
            SNOW (float): Snow fall. Default 0.0
            SDEPTH (float): Sowing depth in cm. Default 3.0 cm
            GDDE (float): Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.
            DSGFT (float): GDD from End Ear Growth to Start Grain Filling period. Default 200 degree-days
            VREQ  (float): Vernalization required for max.development rate (VDays). Default 505 degree-days
            PHINT (float): Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. Default 95.0 degree-days
            P1V (float): Development genetic coefficients, vernalization. 1 for spring type, 5 for winter type. Default 4.85
            P1D (float): Development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length). Default 3.675
            P5 (float): Grain filling degree days. Old value was divided by 10. Default 500 degree-days.
            P6 (float): Approximate the thermal time from physiological maturity to harvest. Default 250.
            DAYS_GERMIMATION_LIMIT (float): Threshold for days to germination. Default 40
            TT_EMERGENCE_LIMIT (int): Threshold for thermal time to emergence. Default 300 degree-days
            TT_TDU_LIMIT (float): Threshold for thermal development units (TDU). Default 400  degree-days
            ADAH (int): Number of days after heading. A threshold used for anthesis date after planting. Default is 6 days after heading.
            
        Returns:
            growstages (dict): A dictionary with all phenological stages and addtional useful information
            
    '''
    if (initparams is None):
        print("Please check out the input parameters")
        return
    
    # Initialization of variables 
    params = dict(
        weather = None, # Weather data of the site
        sowing_date = "", # Sowing date in YYYY-MM-DD
        latitude = -90.0, # Latitude of the site
        longitude = -180.0, # Longitude of the site
        genotype = "", # Name of the grand parent in IWIN pedigrees database 
        TT_TBASE = 0.0, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = 26, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = 34, # Thermal time maximum temperature
        CIVIL_TWILIGHT = 0.0, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        HI = 0.0, # Hardiness Index
        SNOW = 0, # Snow fall
        SDEPTH = 3.0, # Sowing depth in cm
        GDDE = 6.2, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = 200, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = 505.0, # Vernalization required for max.development rate (VDays)
        PHINT = 95.0, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = 1.0, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = 3.675, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = 500, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = 250, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = 40, # threshold for days to germination
        TT_EMERGENCE_LIMIT = 300, # threshold for thermal time to emergence
        TT_TDU_LIMIT = 400, # threshold for thermal development units (TDU)
        ADAH = 6, # threshold for anthesis date after planting. This is a 6 days after heading.
    )
    if (initparams is not None):
        params = {**params, **initparams}
    
    # Validate
    if (params['sowing_date']=="" or params['sowing_date'] is None):
        print("Sowing date not defined")
        return
    if (params['latitude']==-90.0 or params['latitude'] is None):
        print("Problem with location of the site. Check the geographic coordinates.")
        return
    if (params['weather'] is None):
        print("Weather data is not available")
        return
    else:
        weather = params['weather']
    
    # ---------------------
    # GDD limits
    # ---------------------
    #P3 = params['PHINT'] * 2
    #P4 = params['DSGFT'] #200 # APSIM-Wheat = 120 # GDD from End Ear Growth to Start Grain Filling period
    
    growstages = {
            '7': {'istage_old': 'Sowing', 'istage': 'Fallow', 'desc': 'No crop present to Sowing', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '8': {'istage_old': 'Germinate', 'istage': 'Sowing', 'desc': 'Sowing to Germination', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '9': {'istage_old': 'Emergence', 'istage': 'Germinate', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '1': {'istage_old': 'Term Spklt', 'istage': 'Emergence', 'desc': 'Emergence to End of Juvenile', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '2': {'istage_old': 'End Veg', 'istage': 'End Juveni', 'desc': 'End of Juvenile to End of Vegetative growth', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '2.5': {'istage_old': 'Anthesis', 'istage': 'Anthesis', 'desc': 'Anthesis', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '3': {'istage_old': 'End Ear Gr', 'istage': 'End Veg', 'desc': 'End of Vegetative Growth to End of Ear Grow', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '4': {'istage_old': 'Beg Gr Fil', 'istage': 'End Ear Gr', 'desc': 'End of Ear Growth to Start of Grain Filling', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            '5': {'istage_old': 'End Gr Fil', 'istage': 'Beg Gr Fil', 'desc': 'Start of Grain Filling to Maturity', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''},
            #'6': {'istage_old': 'Harvest', 'istage': 'Maturity', 'desc': 'End Gr Fil', 'date':'', 'DOY':'', 'AGE':'', 'DAP':'', 'SUMDTT':''}
    }
    
    class StageFailed(Exception):
        def __init__(self, m, istage, err):
            self.message = m
            self.istage = istage
            self.err = err
        def __str__(self):
            return self.message + f" Stage ({self.istage}) - " + f"Error: {self.err}"

    # --------------------------------------------------------------------------
    # DETERMINE SOWING DATE
    # --------------------------------------------------------------------------
    ISTAGE = 7
    try:
        SOWING_DATE = pd.to_datetime(str(params['sowing_date']), format='%Y-%m-%d' )
        DOY = pd.to_datetime(SOWING_DATE).dayofyear

        growstages[f'{ISTAGE}']['date'] = str(SOWING_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(DOY)
        growstages[f'{ISTAGE}']['AGE'] = 0
        growstages[f'{ISTAGE}']['SUMDTT'] = 0
        growstages[f'{ISTAGE}']['DAP'] = 0
        #print("Sowing date:", SOWING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem initializing the determination of phenological stage. Please check your input parameters such as sowing date or latitude of the site", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
        

    # --------------------------------------------------------------------------
    # DETERMINE GERMINATION  DATE
    # --------------------------------------------------------------------------
    ISTAGE = 8
    try:
        SUMDTT = 0.0
        #VF = 0.0
        DAP = 0
        ndays = 1 # Seed germination is a rapid process and is assumed to occur in one day
        w = weather[(weather['DATE']==(SOWING_DATE + pd.DateOffset(days=ndays)) )].reset_index(drop=True)
        GERMINATION_DATE = ''
        Tmin = float(w.iloc[ndays-1]['TMIN'])
        Tmax = float(w.iloc[ndays-1]['TMAX'])
        # Thermal time
        DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                       Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                       Ttop=params['TT_TEMPERATURE_MAXIMUM'])
        SUMDTT = SUMDTT + DTT
        GERMINATION_DATE = w.iloc[ndays-1]['DATE']
        CROP_AGE = str(GERMINATION_DATE - SOWING_DATE).replace(' days 00:00:00','')
        DAP = DAP + int(CROP_AGE)
        growstages[f'{ISTAGE}']['date'] = str(GERMINATION_DATE).split(' ')[0]
        growstages[f'{ISTAGE}']['DOY'] = int(GERMINATION_DATE.dayofyear)
        growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
        growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
        growstages[f'{ISTAGE}']['DAP'] = DAP

        #print("Germination date:", GERMINATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining germination date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return


    # --------------------------------------------------------------------------
    # DETERMINE SEEDLING EMERGENCE DATE
    # --------------------------------------------------------------------------
    ISTAGE = 9
    P9 = 40 + params['GDDE'] * params['SDEPTH']
    try:
        SUMDTT = 0.0
        #print("Growing degree days from germination to emergence (P9): ",P9) 
        # The crop will die if germination has not occurred before a certain period (eg. 40 days)
        
        EMERGENCE_DATE = ''
        w = weather[weather['DATE']>=GERMINATION_DATE].reset_index(drop=True)
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            SUMDTT = SUMDTT + DTT

            if (SUMDTT >= P9 or SUMDTT > params['TT_EMERGENCE_LIMIT']):
                EMERGENCE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(EMERGENCE_DATE - GERMINATION_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(EMERGENCE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(EMERGENCE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(EMERGENCE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break

        #print("Emergence date: ", EMERGENCE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining emergence date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------------------
    # DETERMINE DURATION OF VEGETATIVE PHASE (END JUVENILE DATE - END OF VEGETATION GROWTH
    # --------------------------------------------------------------------------------------
    ISTAGE = 1
    try: 
        isVernalization = True
        SUMDTT = SUMDTT - P9 
        CUMVD = 0
        TDU = 0
        DF = 0.001
        
        w = weather[weather['DATE']>=EMERGENCE_DATE].reset_index(drop=True)
        END_JUVENILE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            if (isVernalization is True):
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                if (CUMVD < params['VREQ']):
                    VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                    if (VF < 0.3):
                        TDU = TDU + DTT * min(VF, DF)
                    else:
                        DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                        TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                        DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                        TDU = TDU + DTT * min(VF, DF)
                    SUMDTT = TDU
                else:
                    isVernalization = False
            else:
                SUMDTT = SUMDTT + DTT

            if (SUMDTT > P9 ): #or SUMDTT > TT_emergence when reached the lower TT
                END_JUVENILE_DATE = w.iloc[i]['DATE']
                CROP_AGE = str(END_JUVENILE_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                DAP = DAP + int(CROP_AGE)
                growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                growstages[f'{ISTAGE}']['date'] = str(END_JUVENILE_DATE).split(' ')[0]
                growstages[f'{ISTAGE}']['DOY'] = int(END_JUVENILE_DATE.dayofyear)
                growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                growstages[f'{ISTAGE}']['DAP'] = DAP
                #print("Thermal time reached at DAP ", i+1, str(END_JUVENILE_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                break
            #if (DTT > params['TT_EMERGENCE_LIMIT']): # TT_EMERGENCE_LIMIT = 300,
            #    # The crop will die if germination has not occurred before a certain period (eg. 40 days or 300oC d)
            #    print("The crop died because emergence has not occurred before {} degree-days".format(params['TT_EMERGENCE_LIMIT']))

        #print("End Juvenile date: ", END_JUVENILE_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of juvenile date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # --------------------------------------------------------------------------
    # DETERMINE END VEGETATION DATE - End of Juvenile to End of Vegetative growth
    # --------------------------------------------------------------------------
    ISTAGE = 1 # <- Note: this must continue with 1 as previous stage (Term Spklt = Emergence to End of Juvenile + End of Juvenile to End of Vegetative growth)
    try:
        isVernalization = True
        VF = 1.0
        w = weather[weather['DATE']>=END_JUVENILE_DATE].reset_index(drop=True)
        END_VEGETATION_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                if (isVernalization is True):
                    Tcmax, Tcmin, Tcrown = crown_temperatures(snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax)
                    CUMVD = vernalization(Tcrown, Tmin, Tmax, CUMVD)
                    if (CUMVD < params['VREQ']):
                        VF = vernalization_factor(P1V=params['P1V'], dV=CUMVD, ISTAGE=ISTAGE)
                        if (VF < 0.3):
                            TDU = TDU + DTT * min(VF, DF)
                        else:
                            DOY = pd.to_datetime(w.iloc[i]['DATE']).dayofyear
                            TWILEN = day_length(DOY=DOY, lat=params['latitude'], p=params['CIVIL_TWILIGHT'])
                            DF = photoperiod_factor(P1D=params['P1D'], day_length=TWILEN)
                            TDU = TDU + DTT * min(VF, DF)
                        SUMDTT = TDU
                    else:
                        isVernalization = False
                else:
                    SUMDTT = SUMDTT + DTT

                # When this reduced thermal time accumulation (TDU) reaches 
                # 400 degree days, Stage 1 development ends
                if (SUMDTT > (params['TT_TDU_LIMIT'] * (params['PHINT'] / 95.0)) ):
                    END_VEGETATION_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_VEGETATION_DATE - END_JUVENILE_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    # Sum of the two phases
                    CROP_AGE_2 = str(END_VEGETATION_DATE - EMERGENCE_DATE).replace(' days 00:00:00','')
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE_2) # Sum of the last two phases
                    growstages[f'{ISTAGE}']['date'] = str(END_VEGETATION_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_VEGETATION_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("End of Juvenile: Thermal time reached at days duration ", i+1,
                    #          str(END_VEGETATION_DATE), CROP_AGE, DAP, round(SUMDTT, 1))
                    break
        else:
            print("Error reading weather data for vegetation phase")

        # print("End of Vegeation Growth ", END_VEGETATION_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of vegetation growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF EAR GROWTH - End of Vegetative Growth to End of Ear Grow (End leaf growth)
    #-----------------------------------------------------------------------------------------------
    ISTAGE = 2 # Terminal spikelet initiation to the end of leaf growth - CERES Stage 2
    try:
        SUMDTT = 0.0
        P2 = params['PHINT'] * 3

        w = weather[weather['DATE']>=END_VEGETATION_DATE].reset_index(drop=True)
        END_OF_EAR_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P2):
                    END_OF_EAR_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_EAR_GROWTH_DATE - END_VEGETATION_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_EAR_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_EAR_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_EAR_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Ear growth",END_OF_EAR_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of ear growth date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    #
    # ----------------------------------------------------------------------------------------------
    # DETERMINE ANTHESIS
    # ----------------------------------------------------------------------------------------------
    # Anthesis date was estimated as occurring 7 d after heading. (based on McMaster and Smika, 1988; McMaster and Wilhelm, 2003; G. S. McMaster, unpubl. data)
    # Here we used 6 days according to IWIN reported anthesis
    ISTAGE = 2.5
    ADAH = params['ADAH']
    CROP_AGE = DAP + ADAH
    ANTHESIS_DATE = END_OF_EAR_GROWTH_DATE + pd.DateOffset(days=ADAH)
    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
    growstages[f'{ISTAGE}']['date'] = str(ANTHESIS_DATE).split(' ')[0]
    growstages[f'{ISTAGE}']['DOY'] = int(ANTHESIS_DATE.dayofyear)
    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1) # TODOs: Esto debe recalcularse
    growstages[f'{ISTAGE}']['DAP'] = DAP + ADAH
    #
    
    # ----------------------------------------------------------------------------------------------
    # DETERMINE END OF PANNICLE GROWTH - End pannicle growth - End of Ear Growth to Start of Grain Filling
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 3 # Preanthesis ear growth - CERES Stage 3.
    try:
        SUMDTT = 0.0 #SUMDTT - P2
        P3 = params['PHINT'] * 2
        #TBASE=0.0

        w = weather[weather['DATE']>END_OF_EAR_GROWTH_DATE].reset_index(drop=True)
        END_OF_PANNICLE_GROWTH_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT

                if (SUMDTT >= P3):
                    END_OF_PANNICLE_GROWTH_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_OF_PANNICLE_GROWTH_DATE - END_OF_EAR_GROWTH_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_OF_PANNICLE_GROWTH_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_OF_PANNICLE_GROWTH_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = 1
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_OF_PANNICLE_GROWTH_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Pre-Anthesis Ear growth",END_OF_PANNICLE_GROWTH_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of pre-anthesis earh growth date (end of pannicle growth date).", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE BEGIN GRAIN FILLING - Grain fill - Start of Grain Filling to Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 4 # Preanthesis ear growth to the beginning of grain filling - CERES Stage 4.
    try: 
        P4 = params['DSGFT'] #200 GDD # APSIM-Wheat = 120
        SUMDTT = 0.0 #SUMDTT - P3

        w = weather[weather['DATE']>=END_OF_PANNICLE_GROWTH_DATE].reset_index(drop=True)
        BEGIN_GRAIN_FILLING_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT
                if (SUMDTT >= P4):
                    BEGIN_GRAIN_FILLING_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(BEGIN_GRAIN_FILLING_DATE - END_OF_PANNICLE_GROWTH_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(BEGIN_GRAIN_FILLING_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(BEGIN_GRAIN_FILLING_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = 1
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(BEGIN_GRAIN_FILLING_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("Begining of Grain fill",BEGIN_GRAIN_FILLING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining begin of grain fill date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return

    # ----------------------------------------------------------------------------------------------
    # DETERMINE END GRAIN FILLING - Maturity
    # ----------------------------------------------------------------------------------------------
    ISTAGE = 5
    try:
        SUMDTT = 0.0 #SUMDTT - P4
        #P5 = 430 + params['P5'] * 20 # P5 = (0.05 X TT_Maturity) - 21.5. ~500 degree-days
        P5 = params['P5'] # 400 + 5.0 * 20  

        w = weather[weather['DATE']>=BEGIN_GRAIN_FILLING_DATE].reset_index(drop=True)
        END_GRAIN_FILLING_DATE = ''
        if (len(w)>0):
            for i in range(len(w)):
                Tmin = float(w.iloc[i]['TMIN'])
                Tmax = float(w.iloc[i]['TMAX'])
                # Thermal time
                DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                               Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                               Ttop=params['TT_TEMPERATURE_MAXIMUM'])
                SUMDTT = SUMDTT + DTT
                if (SUMDTT >= P5):
                    END_GRAIN_FILLING_DATE = w.iloc[i]['DATE']
                    CROP_AGE = str(END_GRAIN_FILLING_DATE - BEGIN_GRAIN_FILLING_DATE).replace(' days 00:00:00','')
                    DAP = DAP + int(CROP_AGE)
                    growstages[f'{ISTAGE}']['AGE'] = int(CROP_AGE)
                    growstages[f'{ISTAGE}']['date'] = str(END_GRAIN_FILLING_DATE).split(' ')[0]
                    growstages[f'{ISTAGE}']['DOY'] = int(END_GRAIN_FILLING_DATE.dayofyear)
                    growstages[f'{ISTAGE}']['SUMDTT'] = round(SUMDTT, 1)
                    growstages[f'{ISTAGE}']['DAP'] = DAP
                    growstages[f'{ISTAGE}']['status'] = 1
                    #if (verbose is True):
                    #    print("Thermal time reached at days duration ", i+1, str(END_GRAIN_FILLING_DATE), 
                    #          CROP_AGE, DAP, round(SUMDTT, 1))
                    break

        #if (verbose is True):
        #    print("End of Grain filling",END_GRAIN_FILLING_DATE)
    except Exception as err:
        try:
            raise StageFailed("Problem determining end of grain fill date.", ISTAGE, err)
        except StageFailed as x:
            print(x)
            return
    #
    return growstages
# -------------------- END MATURITY

# ----------------------------------







# ==================================

# ----------------------------------
'''
    Wheat phenology model

    Implementation of the Wang-Engel model (Wang and Engel, 1998) for simulating the phenological 
    development of winter wheat based on air temperature records and variety-specific cardinal temperatures. 
    In this exercise we will implement an improved version of the Wang-Engel model as detailed 
    in Streck et al. (2003).

    To better understand the model it is recommended some level of familiarity with wheat phenological stages
    and the Zadoks decimal code (Zadoks et al., 1974).

    Taken from https://soilwater.github.io/pynotes-agriscience/notebooks/wheat_phenology.html


    References:

        - Streck, N.A., Weiss, A., Xue, Q. and Baenziger, P.S., 2003. Improving predictions of developmental stages in winter wheat: a modified Wang and Engel model. Agricultural and Forest Meteorology, 115(3-4), pp.139-150.

        - Wang, E. and Engel, T., 1998. Simulation of phenological development of wheat crops. Agricultural systems, 58(1), pp.1-24.

        - Zadoks, J.C., Chang, T.T. and Konzak, C.F., 1974. A decimal code for the growth stages of cereals. Weed research, 14(6), pp.415-421.


'''
# def wheatstages(Tmax,Tmin,photoperiod,par):
    
#     tavg = np.mean([Tmax,Tmin],axis=0)
#     limVg1 = 0.45; # Limit first vegetative stage 0.4=first hollow stem
#     limVg2 = 1;   # Limit anthesis

#     # Define Lamda functions
#     alphafn = lambda TMIN,TOPT,TMAX: log(2)/log((TMAX-TMIN)/(TOPT-TMIN))
#     betafun = lambda TMIN,TOPT,TMAX,T,ALPHA: max((2*(T-TMIN)**ALPHA * (TOPT-TMIN)**ALPHA - (T-TMIN)**(2*ALPHA))/(TOPT-TMIN)**(2*ALPHA),0)
#     hopp = 17; # Optimal photoperiod. Major, 1980. PHOTOPERIOD RESPONSE CHARACTERISTICS CONTROLLING FLOWERING OF NINE CROP SPECIES 
#     Pc = hopp - 4/par['omega']
#     Pfun = lambda Pp: max(1-exp(-par['chi']*par['omega']*(Pp-Pc)),0)
#     Vfun = lambda VD: (VD**5) / ((par['VDfull']/2)**5 + VD**5)

#     # Define weighing functions
#     alphaVg1 = alphafn(par['TminVg1'],par['ToptVg1'],par['TmaxVg1'])
#     alphaVg2 = alphafn(par['TminVg2'],par['ToptVg2'],par['TmaxVg2'])
#     alphaVn = alphafn(par['TminVn'],par['ToptVn'],par['TmaxVn'])
#     alphaRp = alphafn(par['TminRp'],par['ToptRp'],par['TmaxRp'])

    
#     # Initialize arrays
#     N = len(Tmax)
#     r = np.full(N,np.nan)
#     Tstress = np.full(N,np.nan)
#     VD = np.full(N,np.nan)
#     stage = np.full(N,np.nan)
#     TT = np.full(N,np.nan)
#     TTcum = np.full(N,np.nan)
    
#     # Values at time time t=0
#     r[0] = 0
#     Tstress[0] = 0
#     VD[0] = 0
#     stage[0] = 0
#     TT[0] = thermaltime(Tmax[0],Tmin[0], [par['TminEm'],par['TmaxEm']])
#     TTcum[0] = TT[0]
    
#     for i in range(1,N):

#         if TTcum[i-1] < par['TTemerge']:
#             r[i] = 0.0
#             Tlimits = [par['TminEm'],par['TmaxEm']]
            
#         elif stage[i-1] < limVg1:

#             # Temperature
#             if np.logical_or(tavg[i] < par['TminVg1'], tavg[i] > par['TmaxVg1']):
#                 fT = 0.0
#             else:
#                 fT = betafun(par['TminVg1'],par['ToptVg1'],par['TmaxVg1'],tavg[i],alphaVg1)

#             # Photoperiod
#             fP = Pfun(photoperiod[i])

#             # Vernalization
#             if np.logical_or(tavg[i] < par['TminVn'], tavg[i] > par['TmaxVn']):
#                 VD[i] = 0.0 # Strong control over the initial stages.
#             else:
#                 VD[i] = betafun(par['TminVn'],par['ToptVn'],par['TmaxVn'],tavg[i],alphaVn)

#             fV = Vfun(np.nansum(VD[0:i]))

#             r[i] = max(par['RmaxVg1'] * fT * fP * fV, 0.001) # Calculate development rate when vernalization was triggered

#             # Select cardinal temperatures for current stage
#             Tlimits = [par['TminVg1'],par['TmaxVg1']]
            
#             # Temperature stress index
#             Tstress[i] = 1.0 - fT

#         elif np.logical_and(stage[i-1] >= limVg1, stage[i-1] <= limVg2):

#             # Temperature
#             if np.logical_or(tavg[i] < par['TminVg2'], tavg[i] > par['TmaxVg2']):
#                 fT = 0.0
#             else:
#                 fT = betafun(par['TminVg2'],par['ToptVg2'],par['TmaxVg2'],tavg[i],alphaVg2)

#             # Photoperiod
#             fP = Pfun(photoperiod[i])

#             # Calculate development rate
#             r[i] = par['RmaxVg2'] * fT * fP

#             # Select cardinal temperatures for current stage
#             Tlimits = [par['TminVg2'],par['TmaxVg2']]
            
#             # Temperature stress index
#             Tstress[i] = 1.0 - fT

#         else:
#             # Temperature
#             if np.logical_or(tavg[i] < par['TminRp'], tavg[i] > par['TmaxRp']):
#                 fT = 0.0 
#             else:
#                 fT = betafun(par['TminRp'],par['ToptRp'],par['TmaxRp'],tavg[i],alphaRp)

#             # Calculate development rate
#             r[i] = par['RmaxRp'] * fT

#             # Select cardinal temperatures for current stage
#             Tlimits = [par['TminRp'],par['TmaxRp']]
            
#             # Temperature stress index
#             Tstress[i] = 1.0 - fT
        
#         TT[i] = thermaltime(Tmax[i],Tmin[i],Tlimits)
#         TTcum[i] = TTcum[i-1] + TT[i]
#         stage[i] = min(sum(r[0:i]),2)
        
#     return stage,TT,TTcum,Tstress

# ----------------------------------


# ----------------------------------
# ----------------------------------
