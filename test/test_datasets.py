# -*- coding: utf-8 -*-
"""
TestUnit for gridsearch in pottok
"""
import unittest

import os

from pottok import datasets


class TestPottokImg(unittest.TestCase):
    def check_images_exist(self):
        a,b = datasets.load_pottoks(return_array=False)
        assert(os.path.exists(a)==True)
        assert(os.path.exists(b)==True)
    def load_images(self):
        
        a,b = datasets.load_pottoks()
        # check if 3 dim : X,Y,Z
        assert(a.ndim == 3)
        assert(b.ndim == 3)
        
        # check consistency number of pixels
        
        assert(a.size == 443520)
        assert(b.size == 810000)