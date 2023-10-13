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

import os, sys
import math
import pandas as pd
import datetime as dt

from ..utils import drawPhenology

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

def vernalization_factor(P1V=4.85, dV=50, ISTAGE=1):
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
        P1V = 4.85, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
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
    P9 = 40 + params['GDDE'] * params['SDEPTH'] 
    
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
        P9 = 40 + params['GDDE'] * params['SDEPTH'] #
        CUMDTT = 0.0
        SUMDTT = 0.0
        #VF = 0.0
        #print("Growing degree days from germination to emergence (P9): ",P9) # The crop will die if germination has not occurred before a certain period (eg. 40 days)

        w = weather[weather['DATE']>=GERMINATION_DATE].reset_index(drop=True)
        EMERGENCE_DATE = ''
        for i in range(len(w)):
            Tmin = float(w.iloc[i]['TMIN'])
            Tmax = float(w.iloc[i]['TMAX'])
            DTT = thermal_time_calculation( snow_depth=params['SNOW'], Tmin=Tmin, Tmax=Tmax, 
                                           Tbase=params['TT_TBASE'], Topt=params['TT_TEMPERATURE_OPTIMUM'], 
                                           Ttop=params['TT_TEMPERATURE_MAXIMUM'])
            SUMDTT = SUMDTT + DTT

            if (SUMDTT >= P9):
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
