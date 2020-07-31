# -*- coding: utf-8 -*-
# ================================================================
#              _   _        _    
#             | | | |      | |   
#  _ __   ___ | |_| |_ ___ | | __
# | '_ \ / _ \| __| __/ _ \| |/ /
# | |_) | (_) | |_| || (_) |   < 
# | .__/ \___/ \__|\__\___/|_|\_\
# | |                            
# |_|     
# ================================================================
# @author: Olivia Bernardoff, Nicolas Karasiak, Yousra Hamrouni & David Sheeren
# @git: https://github.com/obernardoff/pottok/
# ================================================================
# @author:  Olivia Bernardoff, Nicolas Karasiak, Yousra Hamrouni & David Sheeren

import re

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open('pottok/__init__.py').read()).group(1)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pottok", 
    version=__version__,
    author="Olivia Bernardoff, Nicolas Karasiak, Yousra Hamrouni & David Sheeren",
    author_email="olivia.bernardoff@outlook.fr",
    description="Optimal Transport with raster image, including an optimal transport validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['psutil','numpy','matplotlib','scikit-learn','POT','museotoolbox','museopheno'],
    url="https://github.com/obernardoff/pottok/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.5', #a confirmer 
    zip_safe=False,
    package_data={
      'pottok': ['datasets/blackpottok.tif','datasets/blackpottok.jpg','datasets/brownpottok.jpg','datasets/brownpottok.tif','datasets/brownpottok.gpkg','datasets/blackpottok.gpkg']
   }
)





















