# __main__.py
#******************************************************************************
#
# Estimating Wheat phenology stages using command line interface (CLI)
# 
# version: 0.0.9
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

import os, sys
import logging
from pathlib import Path
import numba as nb
import pandas as pd
import click
import pywheat as pw  
from pywheat.data import load_dataset
from pywheat.pheno import determine_phenology_stage
from pywheat.utils import drawPhenology, readWeatherStationData
from pywheat.data import load_configfiles
#os.environ["NUMBA_DISABLE_JIT"] = '1'
# nb.config.NUMBA_DISABLE_JIT=1
from pywheat.pheno.phenology import run_pheno_cli
import matplotlib.pylab as plt

"""A Click path argument that returns a pathlib Path, not a string"""
class PathType(click.Path):
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# Load configuration
CONFIG_CALIBRATION = load_configfiles()

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    #click.echo(click.style('Welcome to PyWheat library', fg='green'))
    #click.echo("{}".format(pywheat.__version__))
    #click.echo()
    pass

"""Estimate Phenology"""
@cli.command()
@click.option('-lat', '--latitude', type=float, help='Latitude of the site', required=True)
@click.option('-lon', '--longitude', type=float, help='Longitude of the site', required=False)
@click.option('-sd', '--sowing_date', type=str, help='Sowing date of the crop. eg. 1972-03-13', required=True)
@click.option('-tbase', '--tbase', type=float, help='Base temperature for estimate Thermal time. Default 0.0', default=0.0)
@click.option('-tt_topt', '--tt_topt', type=float, help='Thermal time optimum temperature. Default 26', default=26.0)
@click.option('-tt_tmax', '--tt_tmax', type=float, help='Thermal time maximum temperature. Default 34', default=34.0)
@click.option('-sa', '--sunangle', type=float, help='Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0', default=0.0)
# @click.option('-hi', '--HI', type=float, help='HI (float): Hardiness Index. Default 0.0 ', default=0.0)
@click.option('-sn', '--snow', type=float, help='Snow fall. Default 0.0', default=0.0)
@click.option('-sdepth', '--sdepth', type=float, help='Sowing depth in cm. Default 3.0 cm', default=3.0)
@click.option('-gdde', '--gdde', type=float, help='Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.', default=6.2)
@click.option('-dsgft', '--dsgft', type=float, help='GDD from End Ear Growth to Start Grain Filling period. Default 200 degree-days', default=200)
@click.option('-vreq', '--vreq', type=float, help='Vernalization required for max.development rate (VDays). Default 505 degree-days', default=505)
@click.option('-phint', '--phint', type=float, help='Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. Default 95.0 degree-days', default=95.0)
@click.option('-p1v', '--p1v', type=float, help='Development genetic coefficients, vernalization. 1 for spring type, 5 for winter type. Default 4.85', default=4.85)
@click.option('-p1d', '--p1d', type=float, help='Development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length). Default 3.675', default=3.675)
@click.option('-p5', '--p5', type=float, help=' Grain filling degree days. Old value was divided by 10. Default 500 degree-days.', default=500)
@click.option('-p6', '--p6', type=float, help='Approximate the thermal time from physiological maturity to harvest. Default 250.', default=250)
@click.option('-glim', '--glim', type=float, help='Threshold for days to germination. Default 40', default=40)
@click.option('-elim', '--elim', type=float, help='Threshold for thermal time to emergence. Default 300', default=300)
@click.option('-tdu', '--tdu', type=float, help='Threshold for thermal development units (TDU). Default 400 ', default=400)
@click.option('-showfig', '--showfigure', type=bool, help='Display phenological figure in JPG format', default=False)
@click.option('-fmt', '--inputformat', type=str, help='File format of the input weather file. Options CSV, DSSAT .WTH or Parquet', default='csv')
@click.option('-ofmt', '--outputformat', type=str, help='File format of the output phenology file. Options txt or csv', default='txt')
@click.option(
    "-w",
    "--weather",
    type=PathType(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to input weather file in CSV or Parquet format",
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=PathType(file_okay=True, dir_okay=False, writable=True),
    help="Path to output phenology file.",
    required=False,
)
def pheno(latitude: float, longitude: float, sowing_date: str, tbase: float, tt_topt: float, tt_tmax: float, 
          sunangle: float, snow: float, sdepth: float, gdde: float, dsgft: float, vreq: float, phint: float, 
          p1v: float, p1d: float, p5: float, p6: float, glim: float, elim: float, tdu: float, showfigure:False,
          inputformat:str, outputformat:str, weather: Path, output: Path):
    # # Usage: python pywheat pheno -lat 39.0 -sd '1981-10-16' 
    #          -w ./pywheat/data/example/weather_Kansas_State_Univ_WaggerMG_1983.parquet
    #          -showfig True -o ./Test.txt -fmt parquet
    click.echo("Processing phenological phases...")
    # click.echo(click.style('ATTENTION!', blink=True))
    # click.echo(click.style('Some things', reverse=True, fg='cyan'))
    #
    data = None
    if (weather):
        # Load user dataset
        click.echo("Loading user dataset")
        wd = None
        if (inputformat.lower()=='parquet'):
            try:
                wd = pd.read_parquet(weather)
            except Exception as err:
                print("Format of the input weather file seems to be not valid", err)
                sys.exit()
        if (inputformat.lower()=='csv'):
            try:
                wd = pd.read_csv(weather)
            except Exception as err:
                print("Format of the input weather file seems to be not valid", err)
                sys.exit()
        if (inputformat.lower()=='wth'):
            try:
                # eg. /data/example/KSAS.WTH
                wd = readWeatherStationData(weather)
            except Exception as err:
                print("Format of the input weather .WTH file seems to be not valid", err)
                sys.exit()
        else:
            print("File format not valid. Please try .csv or .parquet files")

        data = {
            "Weather":wd
        }
        click.echo(data['Weather'])
    else:
        try:
            click.echo("Loading example dataset")
            data = load_dataset()
            click.echo(data['Weather'])
        except Exception as err:
            print("Problems loading example dataset", err)
            sys.exit()
    click.echo("Setting up parameters")
    params = dict(
        sowing_date = sowing_date, # Sowing date in YYYY-MM-DD
        latitude = latitude, # Latitude of the site
        longitude = longitude, # Longitude of the site
        TT_TBASE = tbase, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = tt_topt, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = tt_tmax, # Thermal time maximum temperature
        CIVIL_TWILIGHT = sunangle, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        # HI = hi, # Hardiness Index
        SNOW = snow, # Snow fall
        SDEPTH = sdepth, # Sowing depth in cm
        GDDE = gdde, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = dsgft, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = vreq, # Vernalization required for max.development rate (VDays)
        PHINT = phint, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = p1v, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = p1d, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = p5, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = p6, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = glim, # threshold for days to germination
        TT_EMERGENCE_LIMIT = elim, # threshold for thermal time to emergence
        TT_TDU_LIMIT = tdu, # threshold for thermal development units (TDU)
    )
    # Processing phenology
    if (data is not None):
        try:
            click.echo("Determining phenological stages")
            growstages = determine_phenology_stage(initparams=params, weather=data['Weather'], 
                                                dispDates=True, dispFigPhenology=showfigure, verbose=False)
            if (showfigure):
                plt.show()
            if (output):
                try:
                    if (outputformat=='txt'):
                        with open(output, "w") as f:
                            f.writelines("RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE \n")
                            for k in growstages.keys():
                                f.writelines("{:4}   {:10} {:>10} {:>4} {:>6} {:>12} {:>12}\n".format(k, growstages[k]['istage_old'], 
                                        growstages[k]['DAP'], growstages[k]['DOY'], growstages[k]['AGE'], 
                                        growstages[k]['SUMDTT'], growstages[k]['date']))
                    elif(outputformat=='csv'):
                        df = pd.DataFrame(growstages).T
                        df.to_csv(output, index=False)
                    logging.info('Phenological stages determined successfully')
                except Exception as err:
                    logging.info('Problem generating output file')
                    print("Problem generating output file")
                    
        except Exception as err:
            logging.info('Problem determining phenological stages')
            print("Problem determining phenological stages", err)
            sys.exit()
    else:
        logging.info('Weather data not found')
        click.echo("Weather data not found")


"""Estimate phenology stages using a faster method"""
@cli.command()
@click.option('-lat', '--latitude', type=float, help='Latitude of the site', required=True)
@click.option('-lon', '--longitude', type=float, help='Longitude of the site', required=False)
@click.option('-sd', '--sowing_date', type=str, help='Sowing date of the crop. eg. 1972-03-13', required=True)
@click.option('-tbase', '--tbase', type=float, help='Base temperature for estimate Thermal time. Default 0.0', default=0.0)
@click.option('-tt_topt', '--tt_topt', type=float, help='Thermal time optimum temperature. Default 26', default=26.0)
@click.option('-tt_tmax', '--tt_tmax', type=float, help='Thermal time maximum temperature. Default 34', default=34.0)
@click.option('-sa', '--sunangle', type=float, help='Sun angle with the horizon. eg. p = 6.0 : civil twilight. Default 0.0', default=0.0)
# @click.option('-hi', '--HI', type=float, help='HI (float): Hardiness Index. Default 0.0 ', default=0.0)
@click.option('-sn', '--snow', type=float, help='Snow fall. Default 0.0', default=0.0)
@click.option('-sdepth', '--sdepth', type=float, help='Sowing depth in cm. Default 3.0 cm', default=3.0)
@click.option('-gdde', '--gdde', type=float, help='Growing degree days per cm seed depth required for emergence, Default 6.2 GDD/cm.', default=6.2)
@click.option('-dsgft', '--dsgft', type=float, help='GDD from End Ear Growth to Start Grain Filling period. Default 200 degree-days', default=200)
@click.option('-vreq', '--vreq', type=float, help='Vernalization required for max.development rate (VDays). Default 505 degree-days', default=505)
@click.option('-phint', '--phint', type=float, help='Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. Default 95.0 degree-days', default=95.0)
@click.option('-p1v', '--p1v', type=float, help='Development genetic coefficients, vernalization. 1 for spring type, 5 for winter type. Default 4.85', default=4.85)
@click.option('-p1d', '--p1d', type=float, help='Development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length). Default 3.675', default=3.675)
@click.option('-p5', '--p5', type=float, help=' Grain filling degree days. Old value was divided by 10. Default 500 degree-days.', default=500)
@click.option('-p6', '--p6', type=float, help='Approximate the thermal time from physiological maturity to harvest. Default 250.', default=250)
@click.option('-glim', '--glim', type=float, help='Threshold for days to germination. Default 40', default=40)
@click.option('-elim', '--elim', type=float, help='Threshold for thermal time to emergence. Default 300', default=300)
@click.option('-tdu', '--tdu', type=float, help='Threshold for thermal development units (TDU). Default 400 ', default=400)
@click.option('-fmt', '--inputformat', type=str, help='File format of the input weather file. Options CSV, DSSAT .WTH or Parquet', default='csv')
@click.option('-ofmt', '--outputformat', type=str, help='File format of the output phenology file. Options txt or csv', default='txt')
@click.option('-best', '--bestmodel', type=bool, help='Use the calibration parameters to estimate phenology', default=True)
@click.option(
    "-w",
    "--weather",
    type=PathType(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to input weather file in CSV or Parquet format",
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=PathType(file_okay=True, dir_okay=False, writable=True),
    help="Path to output phenology file.",
    required=False,
)
@click.option('-verbose', '--verbose', type=bool, help='Display comments', default=True)
def phenology(latitude: float, longitude: float, sowing_date: str, tbase: float, tt_topt: float, tt_tmax: float, 
          sunangle: float, snow: float, sdepth: float, gdde: float, dsgft: float, vreq: float, phint: float, 
          p1v: float, p1d: float, p5: float, p6: float, glim: float, elim: float, tdu: float, 
          inputformat:str, outputformat:str, bestmodel:bool, weather: Path, output: Path, verbose: True):
    # # Usage: python pywheat phenology -lat 39.0 -lon -120.0 -sd '1981-10-16' 
    #          -w ./pywheat/data/example/weather_Kansas_State_Univ_WaggerMG_1983.parquet
    #          -showfig True -o ./Test.txt -fmt parquet
    global CONFIG_CALIBRATION
    if (verbose):
        click.echo("Processing phenological phases...")
    data = None
    if (weather):
        # Load user dataset
        if (verbose):
            click.echo("Loading user dataset")
        wd = None
        if (inputformat.lower()=='parquet'):
            try:
                wd = pd.read_parquet(weather)
                if ('DATE' not in wd.columns):
                    print("Format of the input weather file seems to be not valid", err)
                    sys.exit()
            except Exception as err:
                print("Format of the input weather file seems to be not valid", err)
                sys.exit()
        if (inputformat.lower()=='csv'):
            try:
                wd = pd.read_csv(weather)
                if ('DATE' not in wd.columns):
                    print("Format of the input weather file seems to be not valid", err)
                    sys.exit()
            except Exception as err:
                print("Format of the input weather file seems to be not valid", err)
                sys.exit()
        if (inputformat.lower()=='wth'):
            try:
                # eg. /data/example/KSAS.WTH
                wd = readWeatherStationData(weather)
                if ('DATE' not in wd.columns):
                    print("Format of the input weather file seems to be not valid", err)
                    sys.exit()
            except Exception as err:
                print("Format of the input weather .WTH file seems to be not valid", err)
                sys.exit()
        else:
            print("File format not valid. Please try .csv or .parquet files")

        data = {
            "Weather":wd
        }
        # if (verbose): # Display Weather data table
        #     click.echo(data['Weather'])
    else:
        try:
            if (verbose):
                click.echo("Loading example dataset")
            data = load_dataset()
            click.echo(data['Weather'])
        except Exception as err:
            print("Problems loading example dataset", err)
            sys.exit()
    if (verbose):
        click.echo("Setting up parameters")
    initparams = dict(
        TT_TBASE = tbase, # Base Temperature, 2.0 to estimate HI
        TT_TEMPERATURE_OPTIMUM = tt_topt, # Thermal time optimum temperature
        TT_TEMPERATURE_MAXIMUM = tt_tmax, # Thermal time maximum temperature
        CIVIL_TWILIGHT = sunangle, # Sun angle with the horizon. eg. p = 6.0 : civil twilight,
        # HI = hi, # Hardiness Index
        SNOW = snow, # Snow fall
        SDEPTH = sdepth, # Sowing depth in cm
        GDDE = gdde, # Growing degree days per cm seed depth required for emergence, GDD/cm
        DSGFT = dsgft, # GDD from End Ear Growth to Start Grain Filling period
        VREQ  = vreq, # Vernalization required for max.development rate (VDays)
        PHINT = phint, # Phyllochron. A good estimate for PHINT is 95 degree days. This value for PHINT is appropriate except for spring sown wheat in latitudes greater than 30 degrees north and 30 degrees south, in which cases a value for PHINT of 75 degree days is suggested. 
        P1V = p1v, # development genetic coefficients, vernalization. 1 for spring type, 5 for winter type
        P1D = p1d, # development genetic coefficients, Photoperiod (1 - 6, low- high sensitive to day length)
        P5 = p5, # grain filling degree days eg. 500 degree-days. Old value was divided by 10.
        P6 = p6, # approximate the thermal time from physiological maturity to harvest
        DAYS_GERMIMATION_LIMIT = glim, # threshold for days to germination
        TT_EMERGENCE_LIMIT = elim, # threshold for thermal time to emergence
        TT_TDU_LIMIT = tdu, # threshold for thermal development units (TDU)
    )
    # Processing phenology
    if (data is not None):
        try:
            if (verbose):
                click.echo("Determining phenological stages")
               
            growstages = run_pheno_cli(initparams, sowing_date, latitude, longitude, wd, CONFIG_CALIBRATION, 
                                                 best=bestmodel, fmt=outputformat, output=output)
            
            # if (output):
            #     try:
            #         if (outputformat=='txt'):
            #             with open(output, "w") as f:
            #                 f.writelines("RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE \n")
            #                 for k in growstages.keys():
            #                     f.writelines("{:4}   {:10} {:>10} {:>4} {:>6} {:>12} {:>12}\n".format(k, growstages[k]['istage_old'], 
            #                             growstages[k]['DAP'], growstages[k]['DOY'], growstages[k]['AGE'], 
            #                             growstages[k]['SUMDTT'], growstages[k]['date']))
            #         elif(outputformat=='csv'):
            #             df = pd.DataFrame(growstages).T
            #             df.to_csv(output, index=False)
            #         logging.info('Phenological stages determined successfully')
            #     except Exception as err:
            #         logging.info('Problem generating output file')
            #         print("Problem generating output file")
                    
        except Exception as err:
            logging.info('Problem determining phenological stages')
            print("Problem determining phenological stages", err)
            sys.exit()
    else:
        logging.info('Weather data not found')
        click.echo("Weather data not found")

if __name__ == "__main__":
    # nb.config.NUMBA_DISABLE_JIT=1
    logging.getLogger().setLevel(logging.INFO)
    click.echo(click.style('Welcome to PyWheat library', fg='green'))
    click.echo("{}".format(pw.__version__))
    click.echo()
    cli()