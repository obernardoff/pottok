#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:46:22 2020

@author: enka
"""

import numpy as np
import ot
import pottok
import museotoolbox as mtb

from sklearn.preprocessing import StandardScaler

X,y = mtb.datasets.load_historical_data(True,low_res=True)

otgs = pottok.OptimalTransportGridSearch(params=dict(log=[True,False]),transport_function=ot.da.SinkhornTransport)

otgs.preprocessing(Xs=X.astype(np.float64),ys=y,Xt=X.astype(np.float64),yt=y)

otgs.fit_circular()

###

img_source, vector_source = mtb.datasets.load_historical_data(low_res=True)

rot = pottok.RasterOptimalTransport(params=dict(log=[True,False]),transport_function=ot.da.SinkhornTransport)

rot.preprocessing(image_source = img_source, image_target = img_source, vector_source = vector_source, vector_target = vector_source, label_source = 'Class', label_target = 'Class',scaler=False)

rot.fit_circular()
assert( np.all(rot.predict_transfer(X)  ==  otgs.predict_transfer(X) )
