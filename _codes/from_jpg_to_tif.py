#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 09:18:23 2020

@author: enka
"""

vector = "/home/enka/blackpottok.gpkg"
raster = "/mnt/bigone/cours/Stage/2020/olivia/pottok/pottok/datasets/blackpottok.jpg"
raster = "/tmp/blackpottok.tif"
raster = "/tmp/brownpottok.tif"

import museotoolbox as mtb
rM = mtb.processing.RasterMath(raster)
x = lambda x : x
rM.add_function(x,"/mnt/bigone/cours/Stage/2020/olivia/pottok/pottok/datasets/brownpottok.tif",compress='jpg')
rM.run()

mtb.processing.extract_ROI("/mnt/bigone/cours/Stage/2020/olivia/pottok/pottok/datasets/blackpottok.tif",vector,'level')
