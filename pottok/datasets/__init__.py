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

from museotoolbox.processing import RasterMath,extract_ROI

def load_pottoks(return_X_y=True,return_target=True):
    """
    Load two images of pottoks.
    
    Parameters
    -----------
    return_X_y : bool, optional (default=True)
        If True, will return the array where there are labels (X) and the labels (y)
        If False, will return only X (the array of the image)
    return_array : bool, optional (default=True)
        If True, will return two arrays
        If False, will return path of the images.
    
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
                'brownpottok')
    black_pottok_uri = os.path.join(
                __pathFile,
                'blackpottok')
    
    to_return = []
    if return_X_y:
        Xs,ys = extract_ROI(brown_pottok_uri+'.tif',brown_pottok_uri+'.gpkg','level')
        to_return.extend([Xs,ys])
        
        if return_target: #to have only labels
            Xt,yt = extract_ROI(black_pottok_uri+'.tif',black_pottok_uri+'.gpkg','level')
            to_return.extend([Xt,yt])
        
        
    if return_X_y is False: #for all image and no label
        brown_pottok_arr = [i for i in RasterMath(brown_pottok_uri+'.tif',return_3d=True,block_size=[-1,-1],verbose=False).read_block_per_block()]
        to_return.append(brown_pottok_arr[0].data)
        if return_target:
            black_pottok_arr = [i for i in RasterMath(black_pottok_uri+'.tif',return_3d=True,block_size=[-1,-1],verbose=False).read_block_per_block()]
            to_return.append(black_pottok_arr[0].data)
        
    return to_return