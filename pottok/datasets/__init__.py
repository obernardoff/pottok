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
from museotoolbox.processing import RasterMath, extract_ROI
import os
__pathFile = os.path.dirname(os.path.realpath(__file__))


def load_pottoks(return_X_y=True, return_target=True, return_only_path=False):
    """
    Load two images of pottoks.

    Parameters
    -----------
    return_X_y : bool, optional (default=True)
        If True, will return the array in 2d where there are labels (X) and the labels (y)
        If False, will return only X in 3d (the array of the image)
    return_target : bool, optional (default=True)
        If True, will return two arrays in a list
        If False, will return only the source (a brown pottok)

    Examples
    --------
    >>> brown_pottok_arr, brown_pottok_label = load_pottoks(return_target=False)
    >>> brown_pottok_arr.shape
    (4610, 3)
    >>> brown_pottok_label.shape
    (4610,)
    """
    brown_pottok_uri = os.path.join(
        __pathFile,
        'brownpottok')
    black_pottok_uri = os.path.join(
        __pathFile,
        'blackpottok')

    to_return = []
    
    if return_only_path :
        to_return.extend([brown_pottok_uri + '.tif', brown_pottok_uri + '.gpkg'])  
        if return_target : 
            to_return.extend([black_pottok_uri + '.tif', black_pottok_uri + '.gpkg'])  

        
        
    elif return_X_y:
        Xs, ys = extract_ROI(brown_pottok_uri + '.tif',
                             brown_pottok_uri + '.gpkg', 'level')
        to_return.extend([Xs, ys])
        


        if return_target:
            Xt, yt = extract_ROI(
                black_pottok_uri + '.tif', black_pottok_uri + '.gpkg', 'level')
            to_return.extend([Xt, yt])

    elif return_X_y is False:
        brown_pottok_arr = RasterMath(
            brown_pottok_uri + '.tif',
            return_3d=True,
            verbose=False).get_image_as_array()

        to_return.append(brown_pottok_arr)
        if return_target:
            black_pottok_arr = RasterMath(
                black_pottok_uri + '.tif',
                return_3d=True,
                verbose=False).get_image_as_array()
            to_return.append(black_pottok_arr)

    return to_return
