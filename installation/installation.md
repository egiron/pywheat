<!-- ---
hide:
  - navigation
  #- toc
--- -->
# Installation

PyWheat package is published as a [Python package] and can be installed with
[`pip`][pip] (the Python package manager), ideally by using a [virtual environment]. 

  [pip]: #with-pip
  [Python package]: https://pypi.org/project/pywheat/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment

### Supported Python versions
- Python 3.10 - 3.11
- pip version 19.0 or higher for Linux (requires manylinux2014 support) and Windows. pip version 20.3 or higher for macOS.

### System requirements
- Ubuntu 16.04 or higher (64-bit)
- macOS 10.12.6 (Sierra) or higher (64-bit)
- Windows Native - Windows 7 or higher (64-bit)
- Windows WSL2 - Windows 10 19044 or higher (64-bit)

### Environment <small>optional</small> { #environment }

We recommend using a [virtual environment], which is an isolated Python runtime.
If you are in a virtual environment, any packages that you install or upgrade
will be local to the environment. If you run into problems, you can
just delete and recreate the environment. It's trivial to set up:

-   Create a new virtual environment with:

    ```
    python3 -m venv venv
    ```

-   Activate the environment with:

    === ":material-apple: macOS"

        ``` sh
        . venv/bin/activate
        ```

    === ":fontawesome-brands-windows: Windows"

        ``` sh
        . venv/Scripts/activate
        ```

    === ":material-linux: Linux"

        ``` sh
        . venv/bin/activate
        ```


    Your terminal should now print `(venv)` before the prompt, which is how you
    know that you are inside the virtual environment that you just created.

-   Exit the environment with:

    ```
    deactivate
    ```

  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment


### with pip <small>recommended</small> { #with-pip data-toc-label="with pip" }

Open up a terminal and install PyWheat with:

=== "Stable"

    ``` sh
    pip install pywheat
    ```

    This is the preferred method to install `pywheat`, as it will always install the most recent stable release.

=== "1.x"

    ``` sh
    pip install pywheat=="1.*"
    ```

    or from sources:

    ``` sh
    pip install "git+https://github.com/egiron/pywheat@v1.0.0"
    ```

=== "Latest"

    ``` sh
    pip install git+https://github.com/egiron/pywheat
    ```

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.


This will automatically install compatible versions of all dependencies:
[Numpy], [Numba], [Pandas], [Scikit-learn], [Scipy], [Matplotlib], [Seaborn], [IPython], [Shapely] and [Arrow]. PyWheat library always strives to support the latest versions, so there's no need to install those packages separately.

  [Numpy]: https://numpy.org/
  [Numba]: https://numba.pydata.org/
  [Pandas]: https://pandas.pydata.org/
  [Scikit-learn]: https://scikit-learn.org
  [Scipy]: https://scipy.org/
  [Matplotlib]: https://matplotlib.org/
  [Seaborn]: https://seaborn.pydata.org/
  [IPython]: https://ipython.org/
  [Shapely]: https://shapely.readthedocs.io/en/stable/index.html
  [Arrow]: https://arrow.apache.org/docs/python/index.html


#### Optional dependencies
- [vaex](https://github.com/vaexio/vaex), is a high performance Python library for lazy Out-of-Core DataFrames (similar to Pandas), to visualize and explore big tabular datasets. Vaex uses memory mapping, zero memory copy policy and lazy computations for best performance (no memory wasted). Used to explore big combinations datasets saved in Parquet or HDF5 format.
- [duckdb](https://duckdb.org/), is an in-process SQL OLAP database management system. It provides support for both reading and writing Parquet files in an efficient manner, as well as support for pushing filters and projections into the Parquet file scans.

---

### with git

PyWheat package can be directly used from [GitHub] by cloning the
repository into a subfolder of your project root which might be useful if you
want to use the very latest version.

Use [Git] to clone the [pywheat repository]:
``` sh
git clone https://github.com/egiron/pywheat.git
cd pywheat
```
  [GitHub]: https://github.com/egiron/pywheat/
  [Git]: https://git-scm.com/
  [pywheat repository]: https://github.com/egiron/pywheat


After cloning from `git`, you must install all required dependencies with:

``` sh
pip install -e pywheat
# or
pip install -e .
```

### Verify install
``` sh
python3 -c "import pywheat; print(pywheat.__version__)"
```

If a pywheat version similar to `pywheat version 1.0.0` is returned, you've installed PyWheat package successfully.

!!! Success  "Success: PyWheat is now installed."

!!! note "Support"

    If you're using PyWheat library in your organization and need
    assistance, e.g., to __reduce processing times__, __improve performance__ or
    ensure compliance, [__get in touch__](mailto:e.giron.e@gmail.com)
    to discuss our __support__ offerings. We're happy to help!

    Bugs may be reported at [Issues](https://github.com/egiron/pywheat/issues).


