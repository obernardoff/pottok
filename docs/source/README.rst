

.. image:: https://github.com/obernardoff/pottok/raw/master/metadata/logopottok.png
   :target: https://github.com/obernardoff/pottok/raw/master/metadata/logopottok.png
   :alt: Pottok logo



.. image:: https://readthedocs.org/projects/pottok/badge/?version=latest
   :target: https://pottok.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status


.. image:: https://badge.fury.io/py/pottok.svg
   :target: https://badge.fury.io/py/pottok
   :alt: PyPI version


.. image:: https://api.travis-ci.com/obernardoff/pottok.svg?branch=master
   :target: https://travis-ci.com/obernardoff/pottok
   :alt: Build status


.. image:: https://pepy.tech/badge/pottok
   :target: https://pepy.tech/project/pottok
   :alt: Downloads


Pottok - Python Optimal Transport for Terrestrial Observation Knowledge
=======================================================================

Pottok is a python library based on POT Python Optimal Transport. It provides a grid search for POT with two different method : crossed and circular.

Classes description
===================

The two avalable classes are :


* pottok.OptimalTransportGridsearch : Create a grid search compatible with POT. The user can use the circular grid search based on the circular validation and crossed grid search based on crossed validation. This one can use scikit learn or museotoolbox GridSearchCV.
* pottok.RasterOptimalTransport : this class is specifically for raster use. 

How it works ?
==============

Since a picture is worth more than a thousand words, we will use a concrete example to illustrate these two grid search

..

   Add tetris draw


Pottok installation
===================
