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


install_requires = []

with open('./requirements.txt') as requirements_txt:
    requirements = requirements_txt.read().strip().splitlines()
    for requirement in requirements:
        if requirement.startswith('#'):
            continue
        elif requirement.startswith('-e '):
            install_requires.append(requirement.split('=')[1])
        else:
            install_requires.append(requirement)


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
    install_requires=install_requires,
    url="https://github.com/obernardoff/pottok/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Optimal Transport",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Remote Sensing"
    ],
    python_requires='>=3.5', #a confirmer 
    zip_safe=False,
    package_data={
      'pottok': ['datasets/blackpottok.jpg','datasets/brownpottok.jpg','datasets/brownpottok.gpkg','datasets/brownpottok.gpkg']
   }
)





















