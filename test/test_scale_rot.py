"""
TestUnit for scaler in pottok
"""
import unittest

import os

import pottok

import museotoolbox as mtb
from sklearn.preprocessing import StandardScaler  # centrer-réduire

source_image,source_vector,target_image,target_vector = pottok.datasets.load_pottoks(return_only_path = True)

brown_pottok,black_pottok = pottok.datasets.load_pottoks(return_X_y=False)
brown_pottok = brown_pottok/255
black_pottok = black_pottok/255

label = 'level' 
group = 'level'

class TestPottokScale(unittest.TestCase):
    def test_scaler(self):

        rot_test = pottok.RasterOptimalTransport()
        rot_test._need_scale=True
        rot_test.scaler=StandardScaler
        
        Xs, ys, group_s, pos_s = mtb.processing.extract_ROI(source_image,
                                                            source_vector,
                                                            label,
                                                            group,
                                                            get_pixel_position=True)  # Xsource ysource

        Xt, yt, group_t, pos_t = mtb.processing.extract_ROI(target_image,
                                                            target_vector,
                                                            label,
                                                            group,
                                                            get_pixel_position=True)  # Xsource ysource

        Xs_non_scale = Xs
        Xt_non_scale = Xt
        
        source_array = mtb.processing.RasterMath(source_image,return_3d=False,
                                      verbose=False).get_image_as_array()
        
        target_array = mtb.processing.RasterMath(target_image,return_3d=False,
                                      verbose=False).get_image_as_array()
        
        
        source_array_test = mtb.processing.RasterMath(source_image,return_3d=True,
                                      verbose=False).get_image_as_array()
        
        target_array_test = mtb.processing.RasterMath(target_image,return_3d=True,
                                      verbose=False).get_image_as_array()
        
        
        rot_test._prefit_image(source_array_test.reshape(*source_array.shape),target_array_test.reshape(*target_array.shape))
        
        source_3d = rot_test.source.reshape(*source_array_test.shape)
        target_3d = rot_test.target.reshape(*target_array_test.shape)


        Xs = rot_test.source_scaler.transform(Xs_non_scale)
        Xt = rot_test.target_scaler.transform(Xt_non_scale)
        
        Xs_test = source_3d[pos_s[:,1].astype(int),pos_s[:,0].astype(int)]
        Xt_test = target_3d[pos_t[:,1].astype(int),pos_t[:,0].astype(int)]
    
        assert((Xs_test == Xs).all())
        assert((Xt_test == Xt).all())