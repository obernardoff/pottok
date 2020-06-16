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
"""
The :mod:`pottok.datasets` module gathers available datasets for testing
`pottok`.
"""
import os
__pathFile = os.path.dirname(os.path.realpath(__file__))

from museotoolbox.processing import RasterMath

def load_pottoks(return_array=True):
    """
    Load two images of pottoks.
    
    Parameters
    -----------
    return_array : bool, optional (default=True)
        If False, will return path of the images.
        If True, will return two arrays
    
    Examples
    --------
    >>> brown_pottok,black_pottok = load_pottoks()
    >>> brown_pottok.shape
    (320, 462, 3)
    >>> black_pottok.shape
    (450, 600, 3)
    
    """
    brown_pottok_uri = os.path.join(
                __pathFile,
                'brownpottok.jpg')
    black_pottok_uri = os.path.join(
                __pathFile,
                'blackpottok.jpg')
    
    if return_array:
        
        brown_pottok_arr = [i for i in RasterMath(brown_pottok_uri,return_3d=True,block_size=[-1,-1],verbose=False).read_block_per_block()]
        black_pottok_arr = [i for i in RasterMath(black_pottok_uri,return_3d=True,block_size=[-1,-1],verbose=False).read_block_per_block()]
        
        return brown_pottok_arr[0].data, black_pottok_arr[0].data
    
    else:
        
        return brown_pottok,black_pottok
    
