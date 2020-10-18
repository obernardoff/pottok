![Pottok logo](https://github.com/obernardoff/pottok/raw/master/metadata/logopottok.png)

[![Documentation status](https://readthedocs.org/projects/pottok/badge/?version=latest)](https://pottok.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pottok.svg)](https://badge.fury.io/py/pottok)
[![Build status](https://api.travis-ci.com/obernardoff/pottok.svg?branch=master)](https://travis-ci.com/obernardoff/pottok)
[![Downloads](https://pepy.tech/badge/pottok)](https://pepy.tech/project/pottok)

# Pottok - Python Optimal Transport for Terrestrial Observation Knowledge

Pottok is a python library based on POT Python Optimal Transport. It provides a grid search for POT with two different method : crossed and circular.

# Classes description

The two avalable classes are :
 - pottok.OptimalTransportGridsearch : Create a grid search compatible with POT. The user can use the circular grid search based on the circular validation and crossed grid search based on crossed validation. This one can use scikit learn or museotoolbox GridSearchCV.
 - pottok.RasterOptimalTransport : this class is specifically for raster use. 

# Pottok installation 

We recommend you to install Pottok via conda as it includes gdal dependency :

```shell
conda install -c conda-forge pottok
```

However, if you prefer to install this library via pip, you need to install first gdal, then :

```shell
python3 -m pip install pottok --user
```

# Who built Pottok ?

I am Olivia Bernardoff, I made pottok during my master degree internship at Dynafor Lab. I worked on remote sensing tree species mapping from space throught dense satellite image time series. I tested optimal transport on these times series to overcome data's lack of information. A special thanks goes to [Nicolas Karasiak](http://www.karasiak.net) who initiated me to the beautiful world of the open-source.
