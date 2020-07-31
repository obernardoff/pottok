#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:46:22 2020

@author: enka
"""

import unittest
import numpy as np
import ot
import pottok
import museotoolbox as mtb

from sklearn.preprocessing import StandardScaler

params = dict(reg_e=[10,1000,10000])
divide_target_by = 2
img_source, vector_source = mtb.datasets.load_historical_data(low_res=True)

class TestOTGS(unittest.TestCase):
    def test_compare_otgs_rot(self):
        X,y = mtb.datasets.load_historical_data(True,low_res=True)
        
        otgs = pottok.OptimalTransportGridSearch(params=params,transport_function=ot.da.SinkhornTransport)
        
        otgs.preprocessing(Xs=X,ys=y,Xt=X/divide_target_by,yt=y)
        
        otgs.fit_circular()
        
        ###
        
        
        
        rot = pottok.RasterOptimalTransport(params=params,transport_function=ot.da.SinkhornTransport)
        rot.preprocessing(image_source = img_source, image_target = img_source, vector_source = vector_source, vector_target = vector_source, label_source = 'Class', label_target = 'Class',scaler=False)
        rot.Xt /= divide_target_by
        rot.fit_circular()
        assert( np.all(rot.predict_transfer(X)  ==  otgs.predict_transfer(X) ) )
        assert(otgs.best_params == rot.best_params)
        assert(otgs.best_score == rot.best_score)
        
        
    def test_scale(self):
        X,y = mtb.datasets.load_historical_data(True,low_res=True)
        
        otgs = pottok.OptimalTransportGridSearch(params=params,transport_function=ot.da.SinkhornTransport)
        
        otgs.preprocessing(Xs=X,ys=y,Xt=X/divide_target_by,yt=y,scaler=StandardScaler)
        
        otgs.fit_circular()
        
        
        rot = pottok.RasterOptimalTransport(params=params,transport_function=ot.da.SinkhornTransport)
        rot.preprocessing(image_source = img_source, image_target = img_source, vector_source = vector_source, vector_target = vector_source, label_source = 'Class', label_target = 'Class',scaler=StandardScaler)
        rot.Xt /= divide_target_by
        rot.fit_circular()
        
        # scaler is made for otgs on X only
        # scaler with rot is made on the whole image
        # so scores with scaler are not the same 
        assert(rot.best_score != otgs.best_score)
        
        
