<p align="center">
  <a href="https://egiron.github.io/pywheat/">
    <img src="https://raw.githubusercontent.com/egiron/pywheat/master/docs/assets/logo_iwin2.png"  alt="IWIN tools library">
  </a>
</p>

<p align="center"><h1 align="center">Welcome to PyWheat</h1></p>

<p align="center">
  <!-- <a href="https://orderseed.cimmyt.org/iwin-results.php"><img
    src="https://img.shields.io/badge/CIMMYT-IWIN-blue"
    alt="CIMMYT IWIN"
  /></a> -->
  <a href="https://github.com/egiron/pywheat/actions"><img
    src="https://github.com/egiron/pywheat/actions/workflows/ci.yaml/badge.svg"
    alt="Build"
  /></a>
  <a href="https://pypistats.org/packages/pywheat"><img
    src="https://img.shields.io/pypi/dm/pywheat.svg" 
    alt="Downloads"
  /></a>
  <a href="https://pypi.org/project/pywheat"><img 
    src="https://img.shields.io/pypi/v/pywheat.svg" 
    alt="Python Package Index"
  /></a>
  <a href="https://opensource.org/licenses/"><img 
    src="https://img.shields.io/badge/License-GPL%20v3-yellow.svg" 
    alt="GPLv3 License"
  /></a>
  
</p>

Python library for simulation of wheat phenological development, crop growth and yield at large scales.

# Intro

The PyWheat simulates the wheat growth and development in a daily time-step at field, local, regional and global scales. Most of the algorithms are based on the original Fortran routines of the CERES-Wheat 2.0[^1]. 

To accurately simulate wheat growth, development, and yield, the model takes into account the following processes:

* Phenological development, especially as it is affected by genetics and weather.
* Extension growth of leaves, stems, and roots.
* Biomass accumulation and partitioning, especially reproductive organs.
* Soil water balance and water use by the crop.
<!-- * Soil nitrogen transformations, uptake by the crop, and partitioning among plant parts.  -->

## Features

* Spatial crop modeling at scale
* GIS and remote sensing supported
* Weather data directly from AgERA5 products
* Wheat Mega-environments
* Yield Forecasting
* Risk management 
* Among others



## Quick start

The package for estimating wheat grain yield using pywheat can be installed with `pip`:

``` sh
pip install pywheat
```

This will add a command-line interface (CLI) that you can then use like so:
``` sh
pywheat
```
This simple command shows the following message:
``` sh
Usage: pywheat [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  pheno
  phenology

```

You can see two functions or commands you will be able to run. Use the help option (-h) to see how to proceed.

``` sh
pywheat phenology -h
```

```
Usage: pywheat phenology [OPTIONS]

Options:
  -lat, --latitude FLOAT       Latitude of the site  [required]
  -lon, --longitude FLOAT      Longitude of the site
  -sd, --sowing_date TEXT      Sowing date of the crop. eg. 1972-03-13
                               [required]
  -tbase, --tbase FLOAT        Base temperature for estimate Thermal time.
                               Default 0.0
  -tt_topt, --tt_topt FLOAT    Thermal time optimum temperature. Default 26
  -tt_tmax, --tt_tmax FLOAT    Thermal time maximum temperature. Default 34
  -sa, --sunangle FLOAT        Sun angle with the horizon. eg. p = 6.0 : civil
                               twilight. Default 0.0
  -sn, --snow FLOAT            Snow fall. Default 0.0
  -sdepth, --sdepth FLOAT      Sowing depth in cm. Default 3.0 cm
  -gdde, --gdde FLOAT          Growing degree days per cm seed depth required
                               for emergence, Default 6.2 GDD/cm.
  -dsgft, --dsgft FLOAT        GDD from End Ear Growth to Start Grain Filling
                               period. Default 200 degree-days
  -vreq, --vreq FLOAT          Vernalization required for max.development rate
                               (VDays). Default 505 degree-days
  -phint, --phint FLOAT        Phyllochron. A good estimate for PHINT is 95
                               degree days. This value for PHINT is
                               appropriate except for spring sown wheat in
                               latitudes greater than 30 degrees north and 30
                               degrees south, in which cases a value for PHINT
                               of 75 degree days is suggested. Default 95.0
                               degree-days
  -p1v, --p1v FLOAT            Development genetic coefficients,
                               vernalization. 1 for spring type, 5 for winter
                               type. Default 4.85
  -p1d, --p1d FLOAT            Development genetic coefficients, Photoperiod
                               (1 - 6, low- high sensitive to day length).
                               Default 3.675
  -p5, --p5 FLOAT              Grain filling degree days. Old value was
                               divided by 10. Default 500 degree-days.
  -p6, --p6 FLOAT              Approximate the thermal time from physiological
                               maturity to harvest. Default 250.
  -glim, --glim FLOAT          Threshold for days to germination. Default 40
  -elim, --elim FLOAT          Threshold for thermal time to emergence.
                               Default 300
  -tdu, --tdu FLOAT            Threshold for thermal development units (TDU).
                               Default 400
  -fmt, --inputformat TEXT     File format of the input weather file. Options
                               CSV, DSSAT .WTH or Parquet
  -ofmt, --outputformat TEXT   File format of the output phenology file.
                               Options txt or csv
  -best, --bestmodel BOOLEAN   Use the calibration parameters to estimate
                               phenology
  -w, --weather FILE           Path to input weather file in CSV or Parquet
                               format  [required]
  -o, --output FILE            Path to output phenology file.
  -verbose, --verbose BOOLEAN  Display comments
  -h, --help                   Show this message and exit.
```

### Usage in CLI
``` sh
pywheat phenology -lat 37.18 -lon -99.75 -sd '1981-10-16' \
 -w ./pywheat/data/example/KSAS.WTH -fmt wth -o ./outputs.txt -verbose False
```
!!! Note
    The above instruction use 3 variables (_latitude, longitude and sowing date of the site_) to run the phenology model. It also requires the path of the weather data file in this case in DSSAT format. This will take a minute or so at the first time to compile the main functions and save them to the cache. Next time will be much faster.

```
RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE 
7      Sowing              0  289      0            0   1981-10-16
8      Germinate           1  290      1           28   1981-10-17
9      Emergence           5  294      5           66   1981-10-21
1      Term Spklt         36  325     31         1683   1981-11-21
2      End Veg           147   71    111          290   1982-03-12
3      End Ear Gr        169   93     22          202   1982-04-03
4      Beg Gr Fil        184  108     15          158   1982-04-18
5      End Gr Fil        214  138     30          459   1982-05-18
6      Harvest           228  152     14          267   1982-06-01
```

### Usage in a python session
```
>>> import pywheat
>>> print(pywheat.__version__)
PyWheat version 0.0.9
>>> from pywheat.data import load_dataset
>>> # Load Kansas data
>>> data = load_dataset()
Loading example weather dataset 
from Kansas State University (Wagger,M.G. 1983) stored in DSSAT v4.8.
>>> data['Weather']
          DATE  SRAD  TMAX  TMIN  RAIN
0   1981-10-01  18.9  23.3  10.0   0.0
1   1981-10-02  18.2  22.2   5.6   0.0
2   1981-10-03   2.4  16.7  11.1   0.3
3   1981-10-04  13.8  26.1  12.8  34.0
4   1981-10-05  12.1  26.7  15.6   0.0
..         ...   ...   ...   ...   ...
299 1982-07-27  19.9  31.7  22.2   0.0
300 1982-07-28  24.9  29.4  20.0   0.0
301 1982-07-29  20.9  30.0  16.7   0.0
302 1982-07-30  26.4  30.0  18.9   0.0
303 1982-07-31  26.5  33.3  14.4   0.0

[304 rows x 5 columns]
>>> # Initialization of variables 
>>> # Default variables are commented
>>> params = dict(
...     sowing_date = "1981-10-16", # Sowing date in YYYY-MM-DD
...     latitude = 39.0, # Latitude of the site
... )
>>> from pywheat.pheno import determine_phenology_stage
>>> import matplotlib.pylab as plt
>>> growstages = determine_phenology_stage(initparams=params, weather=data['Weather'], 
...                                        dispDates=True, dispFigPhenology=True, verbose=False)
RSTG   GROWTH STAGE      DAP  DOY   CROP AGE   SUMDTT   DATE 
7      Sowing              0  289      0            0   1981-10-16
8      Germinate           1  290      1           28   1981-10-17
9      Emergence           5  294      5           66   1981-10-21
1      Term Spklt         23  312     18          503   1981-11-08
2      End Veg           122   46     99          286   1982-02-15
3      End Ear Gr        153   77     31          192   1982-03-18
4      Beg Gr Fil        173   97     20          156   1982-04-07
5      End Gr Fil        207  131     34          468   1982-05-11
6      Harvest           221  145     14          264   1982-05-25

# Display a figure with the phenological stages 
plt.show()
```

For detailed installation instructions visit [installation]

For detailed instructions of how-to get started, configuration options, and a demo, visit [Getting Started]

  [installation]: installation.md
  [Getting Started]: getting_started.md


## Upgrade pywheat

If you have installed **pywheat** before and want to upgrade to the latest version, you can run the following command in your terminal:

```sh
pip install -U pywheat
```

<!-- If you use conda, you can update pywheat to the latest version by running the following command in your terminal:

```sh
conda update -c conda-forge pywheat
``` -->

## Feedback

If you have any feedback, please reach out to us at [Feedback](mailto://e.giron.e@gmail.com)


## FAQ

Please read out [frequently asked questions](faq.md) before you send an email.

## Authors

- [@egiron](https://www.github.com/egiron)


## License

**MIT License**

Copyright (c) 2023 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.


**Additional License** please check out [License](license.md)

<p align="center"><strong>Sponsors</strong></p>
<p align="center">
  <a href="https://www.cimmyt.org/" target=_blank><img
    src="assets/logoCIMMYT_letters.png" height="auto" width="200"
  /></a>
</p>
<p>&nbsp;</p>


  [^1]: CERES-Wheat version 2.0 by Dr. Joe T. Ritchie and Dr. Doug Godwin. https://nowlin.css.msu.edu/wheat_book/
  <!-- [^2]: DSSAT. https://dssat.net/
  [^3]: The Agricultural Production Systems sIMulator (APSIM). https://www.apsim.info/ -->
  [^4]: Ritchie, J.T.1991. Wheat phasic development. p. 31-54. In Hanks and Ritchie (ed.) Modeling plant and soil systems. Agron. Monogr. 31, ASA, CSSSA, SSSA, Madison, WI. 
  [^5]: Ritchie, J.T. and D.S. NeSmith. 1991. Temperature and Crop Development. p. 5-29. In Hanks and Ritchie (ed.) Modeling plant and soil systems. Agron. Monogr. 31, ASA, CSSSA, SSSA, Madison, WI. 
  