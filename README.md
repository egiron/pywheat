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

Python library for simulation of wheat phenological development, crop growth and yield.

# Intro

The PyWheat simulates the wheat growth and development in a daily time-step. Most of the algorithms are based on the original routines of the CERES-Wheat 2.0[^1]. PyWheat model has also much in common with the N-Wheat model. Improvements of both models carried out in DSSAT 4.7.5.11 [^2] and APSIM-Wheat 7.5 R3008[^3] are included gradually in this beta version.

To accurately simulate wheat growth, development, and yield, the model takes into account the following processes:

* Phenological development, especially as it is affected by genetics and weather.
* Extension growth of leaves, stems, and roots.
* Biomass accumulation and partitioning, especially reproductive organs.
* Soil water balance and water use by the crop.
* Soil nitrogen transformations, uptake by the crop, and partitioning among plant parts. 


## Quick start

The package for estimating wheat grain yield using pywheat can be installed with `pip`:

``` sh
pip install pywheat
```

For detailed installation instructions visit [installation]

For detailed instructions of how-to get started, configuration options, and a demo, visit [Getting Started]

  [installation]: installation.md
  [Getting Started]: getting_started.md



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
    src="./docs/assets/logoCIMMYT_letters.png" height="auto" width="200"
  /></a>
</p>
<p>&nbsp;</p>


  [^1]: CERES-Wheat version 2.0 by Dr. Joe T. Ritchie and Dr. Doug Godwin. https://nowlin.css.msu.edu/wheat_book/

  [^2]: DSSAT. https://dssat.net/

  [^3]: The Agricultural Production Systems sIMulator (APSIM). https://www.apsim.info/

  [^4]: Ritchie, J.T.1991. Wheat phasic development. p. 31-54. In Hanks and Ritchie (ed.) Modeling plant and soil systems. Agron. Monogr. 31, ASA, CSSSA, SSSA, Madison, WI. 
  
  [^5]: Ritchie, J.T. and D.S. NeSmith. 1991. Temperature and Crop Development. p. 5-29. In Hanks and Ritchie (ed.) Modeling plant and soil systems. Agron. Monogr. 31, ASA, CSSSA, SSSA, Madison, WI. 
 