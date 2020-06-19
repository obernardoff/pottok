# -*- coding: utf-8 -*-
"""
TestUnit for gridsearch in pottok
"""
import unittest

import numpy as np
import matplotlib.pylab as pl
import ot

from pottok import OptimalTransportGridSearch

##############################################################################
# Generate data
# -------------

n_source_samples = 100
n_target_samples = 100
theta = 2 * np.pi / 20
noise_level = 0.1

Xs, ys = ot.datasets.make_data_classif(
    'gaussrot', n_source_samples, nz=noise_level)
Xs_new, _ = ot.datasets.make_data_classif(
    'gaussrot', n_source_samples, nz=noise_level)
Xt, yt = ot.datasets.make_data_classif(
    'gaussrot', n_target_samples, theta=theta, nz=noise_level)

# one of the target mode changes its variance (no linear mapping)
Xt[yt == 2] *= 3
Xt = Xt + 4

class TestOTGS(unittest.TestCase):
    def test_with_image(self):
        
        ##############################################################################
        # Instantiate the different transport algorithms and fit them
        # -----------------------------------------------------------
        
        # MappingTransport with linear kernel
        ot_mapping_linear = ot.da.MappingTransport(
            kernel="linear", mu=1e0, eta=1e-8, bias=True,
            max_iter=20, verbose=True)
        
        ot_mapping_linear.fit(Xs=Xs, Xt=Xt)
        
        # for original source samples, transform applies barycentric mapping
        transp_Xs_linear = ot_mapping_linear.transform(Xs=Xs)
        
        # for out of source samples, transform applies the linear mapping
        transp_Xs_linear_new = ot_mapping_linear.transform(Xs=Xs_new)
        
        trans_grid = OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                            params=dict(kernel="linear", mu=1e0, eta=1e-8, bias=True,max_iter=20, verbose=True))
        
        
        trans_grid.preprocessing(Xs,ys,Xt,yt,scaler = False)
        trans_grid.fit_circular()
        transp_Xs_linear_grid = trans_grid.predict_transfer(Xs)
        transp_Xs_linear_grid_new = trans_grid.predict_transfer(Xs_new)
        
        assert( np.all(transp_Xs_linear_new == transp_Xs_linear_grid_new) )
        assert( np.all(transp_Xs_linear == transp_Xs_linear_grid) )
    
    def Xs_transp_same_with_circular(self):
        
        ##############################################################################
        # Instantiate the different transport algorithms and fit them
        # -----------------------------------------------------------
        
        # MappingTransport with linear kernel
        ot_mapping_linear = ot.da.MappingTransport(
            kernel="linear", mu=1e0, eta=1e-8, bias=True,
            max_iter=20, verbose=True)
        
        ot_mapping_linear.fit(Xs=Xs, Xt=Xt)
        
        # for original source samples, transform applies barycentric mapping
        transp_Xs_linear = ot_mapping_linear.transform(Xs=Xs)
        
        # for out of source samples, transform applies the linear mapping
        transp_Xs_linear_new = ot_mapping_linear.transform(Xs=Xs_new)
        
        trans_grid = OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                            params=dict(kernel="linear", mu=1e0, eta=1e-8, bias=True,max_iter=20, verbose=True))
        
        
        trans_grid.preprocessing(Xs,ys,Xt,yt,scaler = False)
        trans_grid.fit_circular()
        transp_Xs_linear_grid = trans_grid.predict_transfer(Xs)
        transp_Xs_linear_grid_new = trans_grid.predict_transfer(Xs_new)
        
        assert( np.all(transp_Xs_linear_new == transp_Xs_linear_grid_new) )
        assert( np.all(transp_Xs_linear == transp_Xs_linear_grid) )
    
    
    def test_param_grids_sizes(self):
        
        trans_grid = OptimalTransportGridSearch(transport_function = ot.da.SinkhornTransport,
                        params=dict(reg_e=[1e0,1e-1], max_iter=[10], verbose=True))
        trans_grid.preprocessing(Xs,ys,Xt,yt,scaler = False)
        trans_grid.fit_circular()
        
        # score_no_scaled = trans_grid.best_score
        
        # trans_grid = OptimalTransportGridSearch(transport_function = ot.da.SinkhornTransport,
        #                 params=dict(reg_e=[1e0,1e-1], max_iter=10, verbose=True))
        # trans_grid.preprocessing(Xs,ys,Xt,yt, scaler=StandardScaler)
        # trans_grid.fit_circular()
        # score_scaled = trans_grid.best_score
        
        # assert(score_scaled < score_no_scaled)
        
        assert(len(trans_grid.param_grids) == 2) # from gaussian and linear
        
        trans_grid = OptimalTransportGridSearch(transport_function = ot.da.SinkhornTransport,
                        params=dict(reg_e=[1e0,1e-1], max_iter=[10,20], verbose=True))
        trans_grid.preprocessing(Xs,ys,Xt,yt)
        trans_grid.fit_circular()
        assert(len(trans_grid.param_grids) == 4)
