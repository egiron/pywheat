# coding=utf-8

#******************************************************************************
#
# Utils
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

import matplotlib
from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec
#from matplotlib.collections import PathCollection
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns


'''
    Usage: Weather["date"] = Weather.apply(zfilldate, axis=1)

'''
def zfilldate(df):
    y = str(df['Year'])
    m = str(df["Month"])
    d = str(df["Day"])
    ymd =  y + m.zfill(2) + d.zfill(2)
    ymd = dt.datetime.strptime(ymd, "%Y%m%d")
    return ymd #.strftime('%Y-%m-%d')


# Get Datetime for DATE columns
def getFullDate(YrDOY):
    dtime = dt.datetime.strptime(YrDOY, '%y%j')
    return dtime

# Read weather file .WTH
def readWeatherStationData(input_file=None):
    '''Read DSSAT weather file for the experimental station'''
    if (input_file is None):
        #input_file = os.path.join(DSSAT_PATH, 'Weather', "{}.WTH".format(params['filex']))
        print("Path of the weather .WTH file not found.")
        return
    if (os.path.exists(input_file)):
        header = []
        rows = []
        with open(input_file, 'r', encoding='cp1252' ) as f:
            file_pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if ( (len(line.strip())!=0) and (line[0]!= '!') and (line[0]!='*')):
                    if ( line.startswith('@DATE')):
                        header = line.replace('@','').strip().split(' ')
                        header = [x for x in header if x!='']
                    else:
                        values = line.strip().split(' ')
                        values = [x for x in values if x!='']
                        if (len(values)==len(header)):
                            rows.append(values)
        #print(header)
        df = pd.DataFrame(data=rows, columns=header)
        # Get forward data from sowing date 
        df['DATETIME'] = pd.to_datetime(df['DATE'].apply(lambda x: getFullDate(x)), format='%Y-%m-%d' )
        #df = df[df['DATETIME']>=params['plantingDates'][0]]
        # rename columns for pywheat
        df.rename(columns={'DATE':'YYDOY','DATETIME':'DATE'}, inplace=True)
        return df
    else:
        print("Weather file not found")
        

# Read cultivars file WHAPS0XX.CUL for NWheat model
def readCultivarConfig(input_file=None):
    '''Read DSSAT configuration cultivar file for NWheat model'''
    if (input_file is None):
        #input_file = os.path.join(DSSAT_PATH, 'Genotype', 'WHAPS047.CUL')
        print("Path of the cultivar file not found.")
        return
    if (os.path.exists(input_file)):
        header = []
        rows = []
        with open(input_file, 'r', encoding='cp1252' ) as f:
            file_pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if ( (len(line.strip())!=0) and (line[0]!= '!') and (line[0]!='*')):
                    if ( line.startswith('@VAR#')):
                        header = line.replace('@','').replace('#','').replace('VRNAME..........','VRNAME').strip().split(' ')
                        header = [x for x in header if x!='']
                    else:
                        values = line.strip().split(' ')
                        # check out the VRNAME column with name using blank spaces
                        VRNAME = line[7:25].strip()
                        if (len(VRNAME.split(' '))>1):
                            # replace space with underscore
                            VRNAME = VRNAME.replace(' ','_') #.strip()
                            values = (line[0:7] + ' ' + VRNAME + line[25:-1]).strip().split(' ')
                        values = [x for x in values if x!='']
                        if (len(values)==len(header)):
                            rows.append(values)
        #print(header)
        return pd.DataFrame(data=rows, columns=header)
    else:
        print("Cultivar file not found")


# Read ecotypes file WHAPS0XX.ECO for NWheat model
def readEcotypesConfig(input_file=None):
    '''Read DSSAT configuration ecotypes file for NWheat model'''
    if (input_file is None):
        #input_file = os.path.join(DSSAT_PATH, 'Genotype', 'WHAPS047.eco')
        #if (not os.path.exists(input_file)):
        #    input_file = os.path.join(DSSAT_PATH, 'Genotype', 'WHAPS047.ECO')
        print("Path of the ecotypes file not found.")
        return
    if (os.path.exists(input_file)):
        header = []
        rows = []
        with open(input_file, 'r', encoding='cp1252' ) as f:
            file_pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if ( (len(line.strip())!=0) and (line[0]!= '!') and (line[0]!='*')):
                    if ( line.startswith('@ECO#')):
                        header = line.replace('@','').replace('#','').replace('ECONAME.........','ECONAME').strip().split(' ')
                        header = [x for x in header if x!='']
                    else:
                        values = line.strip().split(' ')
                        # check out the ECONAME column with name using blank spaces
                        ECONAME = line[7:25].strip()
                        if (len(ECONAME.split(' '))>1):
                            # replace space with underscore
                            ECONAME = ECONAME.replace(' ','_')
                            values = (line[0:7] + ' ' + ECONAME + line[25:-1]).strip().split(' ')
                        values = [x for x in values if x!='']
                        if (len(values)<len(header)):
                            values.append('') #TSEN
                            values.append('') #CDAY
                        if (len(values)==len(header)):
                            rows.append(values)
        #print(header)
        return pd.DataFrame(data=rows, columns=header)
    else:
        print("Ecotype file not found")
        
#
def readPlantGro(input_file=None, RUN_START=1, RUN_END=2):
    '''Read first run simulation from PlantGro.OUT file
    '''
    if (input_file is None):
        print("Path of the PlantGro.OUT file not found.")
        return
    if (os.path.exists(input_file)):
        StartLine = 0
        EndLine = 1
        cols = []
        rows = []
        with open(input_file, 'r', encoding='cp1252', buffering=100000 ) as f: 
            file_pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                ls = line.split(':')[0].strip()
                if (ls=='*RUN{:>4}'.format(RUN_START)):
                    StartLine = f.tell() #returns the location of the next line
                elif (ls=='*RUN{:>4}'.format(RUN_END)):
                    EndLine = f.tell() 
            # Display lines
            file_pos = f.seek(StartLine)
            l2conv = False
            while file_pos != EndLine:
                line = f.readline()
                if not line:
                    break
                ls = line.split(':')[0].strip()
                if (line.startswith('@YEAR')):
                    pos_tmp = f.tell() 
                    cols = line.replace('@','').strip().split(' ')
                    cols = [x for x in cols if x!='']
                    l2conv = True
                elif (l2conv is True or line.startswith('*DSSAT')): #line.strip()!=' '):
                    values = line.strip().split(' ')
                    values = [x for x in values if x!='']
                    if (len(values)==len(cols)):
                        rows.append(values)
                else:
                    l2conv = False

                file_pos += len(line)
                
        return pd.DataFrame(data=rows, columns=cols)
    else:
        print("PlantGro.OUT file not found!")
#

# --------------------------------------------------
# Draw Phenology
# --------------------------------------------------
def drawPhenology(gs=None, title='Phenological growth phases of Wheat', dpi=150,
                  dispPlants=True, topDAPLabel=True, timeSpanLabel=True, topNameStageLabel=True,
                  topNameStageLabelOpt=True, copyrightLabel=True, saveFig=True, showFig=True, 
                  path_to_save_results='./', fname='Phenological_Phases_Wheat', fmt='jpg'):
    
    df = pd.DataFrame(gs).T
    # Setup parameters for x-axes
    labels = ','.join([str(x) for x in df['istage_old']]).split(',')
    dap_values = ','.join([str(x) for x in df['DAP']]).split(',')
    tt_values = ','.join([str(x) for x in df['SUMDTT']]).split(',')
    date_values = ','.join([str(x) for x in df['date']]).split(',')
    doy_values = ','.join([str(x) for x in df['DOY']]).split(',')
    cropage_values = ','.join([str(x) for x in df['AGE']]).split(',')
    # Convert date strings (e.g. 2014-10-18) to datetime
    dates = list(df['date'])
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    dates_Month_Year = [d.strftime('%b-%Y') for d in dates]
    dates_MMM_DD_YYYY = [d.strftime('%b %d, %Y') for d in dates]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi, constrained_layout=True)

    # -------------------------
    # Figure 1
    # -------------------------
    ax.scatter(dap_values, tt_values, c="#fff") # Don't show the points
    ax.set_title(f'{title}', fontsize=18, y=1.1, pad=16)
    #plt.xticks(rotation=90, fontsize=9)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.grid(visible=False)
    ax.spines['top'].set_linestyle("dashed")
    ax.spines['top'].set_capstyle("butt")
    ax.spines['top'].set_linewidth(.2)  
    ax.spines['bottom'].set_color('black')
    ax.spines[['left',  'right']].set_visible(False)
    # format x-axis with 4-month intervals
    #ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    #plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # ---------------------------
    # ---- Stages x-axis
    def dispXAxes_Phenology(v='Stage', xticklabels=[], centerLabel=False, c='green', ypos_axis=-0.05, 
                            xpos_tile=-0.1, ypos_title=-0.11,  fsize_title=12, fsize_labels=10):
        ax1 = ax.twiny()
        ax1.xaxis.set_ticks_position("bottom")
        ax1.xaxis.set_label_position("bottom")
        ax1.spines["bottom"].set_position(("axes", ypos_axis))
        ax1.spines['bottom'].set_color(c)
        ax1.spines[['left', 'top', 'right']].set_visible(False)
        ax1.set_xticks(np.array(range(0,9)))
        ax1.set_xticklabels(xticklabels, fontsize=fsize_labels)
        ax1.xaxis.label.set_color(c)
        ax1.tick_params(axis='x', colors=c)
        ax1.grid(visible=False)
        if (centerLabel):
            ax1.set_xlabel(f"{v}")
        else:
            ax1.text(xpos_tile, ypos_title, v, transform=ax.transAxes, fontsize=fsize_title, 
                     fontname='Monospace', color=c)

    # Display Stage or Phase name axis
    dispXAxes_Phenology(v='Stage', xticklabels=labels, centerLabel=False, c='green', ypos_axis=-0.05, 
                            xpos_tile=-0.1, ypos_title=-0.11, fsize_title=12, fsize_labels=10)
    # Display Phenology Date axis
    dispXAxes_Phenology(v='Date', xticklabels=dates_MMM_DD_YYYY, centerLabel=False, c='black', ypos_axis=-0.20, 
                            xpos_tile=-0.1, ypos_title=-0.27, fsize_title=12, fsize_labels=8)
    # Display Day of Year axis
    dispXAxes_Phenology(v='DOY', xticklabels=[f"{x}" for x in doy_values], centerLabel=False, 
                        c='purple', ypos_axis=-0.35, xpos_tile=-0.1, ypos_title=-0.43, fsize_title=12, fsize_labels=8)
    # Display Thermal time axis
    dispXAxes_Phenology(v='GDD', xticklabels=[f"{x} °C d" for x in tt_values], centerLabel=False, 
                        c='red', ypos_axis=-0.50, xpos_tile=-0.1, ypos_title=-0.57, fsize_title=12, fsize_labels=8)
    # ---------------------------

    ''' Display the icon of the plant over the figure '''
    def drawPlants(nstage=0, xpos=0.08, ypos=0.398, w=0.04, h=0.04):
        #print(pathlib.Path(__file__).parent.resolve())
        #icons_path = os.path.dirname(os.path.abspath(__file__)) # py2.7 & py3.x
        #icons_path = os.getcwd() +'/pywheat/data/whstages/stage_7.png'
        icons_path = os.path.join(pathlib.Path(__file__).parent.resolve(), '../data/whstages/',f"stage_{nstage}.png")
        img_stage = plt.imread(icons_path)
        newax = fig.add_axes([xpos, ypos, w, h], anchor='C', zorder=1)
        newax.imshow(img_stage)
        newax.axis('off')

    if (dispPlants):
        # ------ Stage 7
        drawPlants(nstage=7, xpos=0.08, ypos=0.398-0.005, w=0.04, h=0.04)
        # ------ Stage 8
        drawPlants(nstage=8, xpos=0.08+0.10, ypos=0.398-0.004, w=0.04, h=0.04)
        # ------ Stage 9
        drawPlants(nstage=9, xpos=0.08+0.20, ypos=0.398-0.002, w=0.055, h=0.055)
        # ------ Stage 1
        drawPlants(nstage=1, xpos=0.08+0.26, ypos=0.398+0.001, w=0.15, h=0.15)
        # ------ Stage 2
        drawPlants(nstage=2, xpos=0.08+0.333, ypos=0.398, w=0.22, h=0.22)
        # ------ Stage 3
        drawPlants(nstage=3, xpos=0.08+0.406, ypos=0.398, w=0.28, h=0.28)
        # ------ Stage 4
        drawPlants(nstage=4, xpos=0.08+0.500, ypos=0.398, w=0.32, h=0.32)
        # ------ Stage 5
        drawPlants(nstage=5, xpos=0.08+0.592, ypos=0.398, w=0.35, h=0.35)
        # ------ Stage 6
        drawPlants(nstage=6, xpos=0.08+0.820, ypos=0.398, w=0.08, h=0.08)
        # ---------------------------
        # Top Label of plants
        if (topDAPLabel):
            topLabel_DAP = [f"{x} days" for x in dap_values]
            fs=8
            clr = '#444'
            ax.text(-0.015, 0.1, topLabel_DAP[0], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.1, 0.1, topLabel_DAP[1], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.23, 0.15, topLabel_DAP[2], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.34, 0.37, topLabel_DAP[3], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.47, 0.56, topLabel_DAP[4], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.59, 0.66, topLabel_DAP[5], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.72, 0.74, topLabel_DAP[6], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.92, 0.74, topLabel_DAP[7], transform=ax.transAxes, fontsize=fs, color=clr)
            ax.text(0.96, 0.16, topLabel_DAP[8], transform=ax.transAxes, fontsize=fs, color=clr)
    # ---------------------------
    if (timeSpanLabel):
        betweenLabel_AGE = [f"{x} d" for x in cropage_values]
        fs=7
        clr = 'darkgray'
        xpos_age = 0.05
        ypos_age = 0.015
        ax.text(xpos_age, ypos_age, betweenLabel_AGE[1], transform=ax.transAxes, fontsize=fs, color=clr)
        ax.text(xpos_age + 0.12, ypos_age, betweenLabel_AGE[2], transform=ax.transAxes, fontsize=fs, color=clr)
        ax.text(xpos_age + 0.245, ypos_age, betweenLabel_AGE[3], transform=ax.transAxes, fontsize=fs, color=clr)
        ax.text(xpos_age + 0.375, ypos_age, betweenLabel_AGE[4], transform=ax.transAxes, fontsize=fs, color=clr)
        ax.text(xpos_age + 0.50, ypos_age, betweenLabel_AGE[5], transform=ax.transAxes, fontsize=fs, color=clr)
        ax.text(xpos_age + 0.62, ypos_age, betweenLabel_AGE[6], transform=ax.transAxes, fontsize=fs, color=clr)
        ax.text(xpos_age + 0.75, ypos_age, betweenLabel_AGE[7], transform=ax.transAxes, fontsize=fs, color=clr)
        ax.text(xpos_age + 0.88, ypos_age, betweenLabel_AGE[8], transform=ax.transAxes, fontsize=fs, color=clr)

    if (topNameStageLabel):
        fs=10
        clr = 'green'
        ypos_phases = 0.94 #1.04
        ax.text(0.25, ypos_phases, 'Vegetative phase', ha='center', transform=ax.transAxes, fontsize=fs, color=clr, alpha=1)
        ax.text(0.60, ypos_phases, 'Reproductive phase', ha='center', transform=ax.transAxes, fontsize=fs, 
                color=clr, alpha=1)
        ax.text(0.84, ypos_phases, 'Grain Filling phase', ha='center', transform=ax.transAxes, fontsize=fs, 
                color=clr, alpha=1)
        # rectangles
        xy, w, h = (0, 0.92), 1, 1
        r = Rectangle(xy, w, h, fc='#f6fef6', ec='gray', lw=0.2, transform=ax.transAxes, zorder=2)
        ax.add_artist(r)

    if (topNameStageLabelOpt):
        fs=10
        clr = 'brown'
        ypos=0.85
        ax.text(0.47, ypos, 'Heading', transform=ax.transAxes, fontsize=fs, color=clr, alpha=0.5)
        ax.text(0.64, ypos, 'Anthesis', transform=ax.transAxes, fontsize=fs, color=clr, alpha=0.5)
        ax.text(0.84, ypos, 'Maturity', transform=ax.transAxes, fontsize=fs, color=clr, alpha=0.5)

    # Add vertical lines
    #vertLines = False
    #if (vertLines):
    #    clr='lightgray'
    #    #ax.axvline(0, ls='--', c=clr, linewidth=0.4, label="Vegetative phase", zorder=-1)
    #    #ax.axvline(2, ls='--', c=clr, linewidth=0.4, zorder=0)
    #    ax.axvline(4, ls='--', c=clr, linewidth=0.4, zorder=-1)
    #    ax.axvline(6, ls='--', c=clr, linewidth=0.4, zorder=-1)

    if (copyrightLabel):
        yr = dt.datetime.now().year
        ax.text(0.88, -0.75, f'Created with PyWheat © {yr}', transform=ax.transAxes, fontsize=7, 
                color='lightgray', zorder=2)

    # ---------------------------
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4) #hspace=0.5, wspace=0.5
    # Save in PDF
    hoy = dt.datetime.now().strftime('%Y%m%d')
    figures_path = path_to_save_results #os.path.join(path_to_save_results, '{}_{}'.format(dirname, hoy))
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    if (saveFig is True and fmt=='pdf'):
        fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                    bbox_inches='tight', orientation='portrait',  
                    edgecolor='none', transparent=False, pad_inches=0.5, dpi=dpi)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=dpi)
    
    if (showFig is True):
        fig.show()
    else:
        del fig
        plt.close();

# ---------------------------------------------
def display_GDD(df, growstages):
    df = df.copy()
    fig, ax1 = plt.subplots(1,1, figsize=(10,6))
    fig.subplots_adjust(top=0.9)
    sns.set_theme(style="whitegrid")
    ax1.scatter(x=df["DAP"].to_numpy(), y=df["AGE"].to_numpy(), label='GDD (Wheat)', color='#fff' )
    #g1 = sns.scatterplot(x="DAP", y="SUMDTT", data=df, label='GDD (Wheat)', 
    #                markers=False, color='red', ax=ax1)
    # 
    # Lines growing dates
    #7 - Fallow - 1997-12-17
    stage_7 = df.iloc[0]['DAP'] #int(df['DAP'][df['date']==growstages['7']['date']])
    #8 - Sowing - 1997-12-19
    stage_8 = df.iloc[1]['DAP'] #int(df['DAP'][df['date']==growstages['8']['date']])
    #9 - Germinate - 1998-01-01
    stage_9 = df.iloc[2]['DAP'] #int(df['DAP'][df['date']==growstages['9']['date']])
    #1 - Emergence - 1998-04-06
    stage_1 = df.iloc[3]['DAP'] #int(df['DAP'][df['date']==growstages['1']['date']])
    #2 - End Juveni - 1998-04-29
    stage_2 = df.iloc[4]['DAP'] #int(df['DAP'][df['date']==growstages['2']['date']])
    #3 - End Veg - 1998-05-14
    stage_3 = df.iloc[5]['DAP'] #int(df['DAP'][df['date']==growstages['3']['date']])
    #4 - End Ear Gr - 1998-05-23
    stage_4 = df.iloc[6]['DAP'] #int(df['DAP'][df['date']==growstages['4']['date']])
    #5 - Beg Gr Fil - 1998-06-18
    stage_5 = df.iloc[7]['DAP'] #int(df['DAP'][df['date']==growstages['5']['date']])
    #6 - Maturity - 
    stage_6 = df.iloc[8]['DAP'] #int(df['DAP'][df['date']==growstages['6']['date']])
    sp = 5
    vp = 1.0
    ax1.axvline(stage_7, ls='--', c='green', linewidth=0.8) #label="Fallow", 
    #ax1.text(0, .2, 'label', fontweight="bold", color='green', ha="left", va="center")
    an7 = ax1.annotate("{} - {}".format(growstages['7']['istage_old'], growstages['7']['date']), (stage_7-sp, vp), c='grey', fontsize=10)
    plt.setp(an7, rotation=90)
    ax1.axvline(stage_8, ls='--', c='green', linewidth=0.8) #label="Sowing", 
    an8 = ax1.annotate("{} - {}".format(growstages['8']['istage_old'], growstages['8']['date']), (stage_8, vp), c='grey', fontsize=10)
    plt.setp(an8, rotation=90)
    ax1.axvline(stage_9, ls='--', c='green', linewidth=0.8) #label="Germinate", 
    an9 = ax1.annotate("{} - {}".format(growstages['9']['istage_old'], growstages['9']['date']), (stage_9+2, vp), c='grey', fontsize=10)
    plt.setp(an9, rotation=90)
    ax1.axvline(stage_1, ls='--', c='green', linewidth=0.8) #label="Emergence", 
    an1 = ax1.annotate("{} - {}".format(growstages['1']['istage_old'], growstages['1']['date']), (stage_1-sp, vp), c='grey', fontsize=10)
    plt.setp(an1, rotation=90)
    ax1.axvline(stage_2, ls='--', c='green', linewidth=0.8) #label="End Juveni",
    an2 = ax1.annotate("{} - {}".format(growstages['2']['istage_old'], growstages['2']['date']), (stage_2-sp, vp), c='grey', fontsize=10)
    plt.setp(an2, rotation=90)
    ax1.axvline(stage_3, ls='--', c='green', linewidth=0.8) #label="End Veg", 
    an3 = ax1.annotate("{} - {}".format(growstages['3']['istage_old'], growstages['3']['date']), (stage_3-sp, vp), c='grey', fontsize=10)
    plt.setp(an3, rotation=90)
    ax1.axvline(stage_4, ls='--', c='green', linewidth=0.8) #label="End Ear Gr", 
    an4 = ax1.annotate("{} - {}".format(growstages['4']['istage_old'], growstages['4']['date']), (stage_4-sp, vp), c='grey', fontsize=10)
    plt.setp(an4, rotation=90)
    ax1.axvline(stage_5, ls='--', c='green', linewidth=0.8) #label="Beg Gr Fil", 
    an5 = ax1.annotate("{} - {}".format(growstages['5']['istage_old'], growstages['5']['date']), (stage_5-sp, vp), c='grey', fontsize=10)
    plt.setp(an5, rotation=90)
    ax1.axvline(stage_6, ls='--', c='green', linewidth=0.8) #label="Maturity", 
    an6 = ax1.annotate("{} - {}".format(growstages['6']['istage_old'], growstages['6']['date']), (stage_6-sp, vp), c='grey', fontsize=10)
    plt.setp(an6, rotation=90)

    plt.xticks(rotation=90)
    ax1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.25)
    #ax1.set_title('NWheat', fontsize=13)
    ax1.set_xlabel('Day after planting (DAP)', fontsize=14)
    #ax1.set_ylabel('Crop age (days)', fontsize=14)
    ax1.set_ylabel('')
    #ax1.legend(loc="best", fontsize=10)
    #ax1.get_legend().remove()
    #ax1.set_xticks([f'{str(x)}' for x in list(df['DAP'])])
    ax1.set_xticks(np.arange(0, df['DAP'].max()+10, 1))  # adjust the y tick frequency
    ax1.set_yticks([])
    
    
    v_lines = []
    v_lines_values = ','.join([str(x) for x in df['DAP'].unique()]).split(',')
    #print(len(ax1.get_xticklabels()))
    for label in ax1.get_xticklabels(): #minor=True
        if str(label.get_text()) in v_lines_values:
            #print(label.get_text())
            if (len(v_lines) < len(v_lines_values)):
                v_lines.append(label.get_position()[0])

    #print(v_lines, v_lines_values)
    #ax1.vlines(v_lines, 0, 1, ls='-.', color='lightgray', linewidth=0.75)
    ax1.set_xticks(v_lines) #ax1.get_xticks()[::len(v_lines_values)])
    ax1.tick_params(axis='x', labelsize=10, color='lightgray', rotation=90)
    # ------------
    
    #fig.suptitle('Sum of growing degree days since stage initiation', fontsize=18) #
    fig.suptitle('Phenological growth stages of Wheat', fontsize=18) #

    #fig.savefig(os.path.join(DATASET_IWIN_PATH, "Wheat_SUMDTT_GrowStages.png"))
    fig.tight_layout()
    plt.show()






