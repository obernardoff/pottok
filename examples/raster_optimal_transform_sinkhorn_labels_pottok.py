# -*- coding: utf-8 -*-
"""
========================================================================
Pottok for image color adaptation with labels - RasterOptimalTransport
========================================================================

Using sinkhorn L1l2
"""

import numpy as np
import matplotlib.pylab as pl
import ot
import pottok
# import os
# __pathFile = os.path.dirname(os.path.realpath(__file__))

##############################################################################
# Load pottoks
# -------------

# Loading X and y


# brown_pottok_uri = os.path.join(
#     __pathFile,
#     'brownpottok')
# black_pottok_uri = os.path.join(
#     __pathFile,
#     'blackpottok')

wdir = "/home/olivia/Bureau/git_clone_stage/pottok/pottok/datasets/"

source_image = str(wdir + 'brownpottok.tif')
target_image = str(wdir + 'blackpottok.tif')
source_vector = str(wdir + 'brownpottok.gpkg' )
target_vector = str(wdir + 'blackpottok.gpkg' )

label = 'level' #le même pour sources et target



##############################################################################
# Optimal transport with SinkhornL1l2 with circular gridsearch
# --------------------------------------------------------------

gridsearch_transport_circular = pottok.RasterOptimalTransport(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
gridsearch_transport_circular.preprocessing(in_image_source = source_image,
                   in_image_target = target_image,
                   in_vector_source = source_vector,
                   in_vector_target = target_vector,
                   in_label_source = label,
                   in_label_target = label,
                   scaler = False)
gridsearch_transport_circular.fit_circular()


# Best grid is {'reg_e': 1.0, 'reg_cl': 0.1}

##############################################################################
# Plot images
# -----------------------

# brown_pottok_transp_circular = gridsearch_transport_circular.predict_transfer(brown_pottok.reshape(-1,3))
# pl.figure(1, figsize=(10,8))

# pl.subplot(2, 2, 1)
# pl.imshow(brown_pottok)
# pl.axis('off')
# pl.title('Brown pottok (Source)')

# pl.subplot(2, 2, 3)
# pl.imshow(black_pottok)
# pl.axis('off')
# pl.title('Black pottok (Target)')

# pl.subplot(2, 2, 4)
# pl.imshow(brown_pottok_transp_circular.reshape(*brown_pottok.shape))
# pl.axis('off')
# pl.title('SinkhornL1l2 (Source to Target with labels)')

# pl.show()




# ##############################################################################
# # Optimal transport with SinkhornL1l2 with crossed gridsearch
# # --------------------------------------------------------------

# gridsearch_transport_crossed = pottok.OptimalTransportGridSearch(transport_function=ot.da.SinkhornL1l2Transport,
#                                         params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
# gridsearch_transport_crossed.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt,scaler=False)
# gridsearch_transport_crossed.fit_crossed()

# # Best grid is {'reg_e': 0.1, 'reg_cl': 0.1}

# ##############################################################################
# # Plot images
# # -----------------------

# brown_pottok_transp_crossed = gridsearch_transport_crossed.predict_transfer(brown_pottok.reshape(-1,3))
# pl.figure(2, figsize=(10,8))

# pl.subplot(2, 2, 1)
# pl.imshow(brown_pottok)
# pl.axis('off')
# pl.title('Brown pottok (Source)')

# pl.subplot(2, 2, 3)
# pl.imshow(black_pottok)
# pl.axis('off')
# pl.title('Black pottok (Target)')

# pl.subplot(2, 2, 4)
# pl.imshow(brown_pottok_transp_crossed.reshape(*brown_pottok.shape))
# pl.axis('off')
# pl.title('SinkhornL1l2 (Source to Target with labels)')

# pl.show()


# ##############################################################################
# # Comparison with pot
# # ----------------------------------


# ot_mapping_linear_circular = ot.da.SinkhornL1l2Transport(
#     reg_e=1.0, reg_cl=0.1,verbose=True)
# ot_mapping_linear_circular.fit(Xs=Xs, ys = ys, Xt=Xt, yt = yt)
# brown_pottok_transp_pot_circular = ot_mapping_linear_circular.transform(brown_pottok.reshape(-1,3))

# if (brown_pottok_transp_pot_circular == brown_pottok_transp_circular).all() : 
#     print ("POT and Pottok give same transformation - circular")
# else : 
#     print ("ERROR : POT and Pottok do not give same transformation")
   
# ot_mapping_linear_crossed = ot.da.SinkhornL1l2Transport(
#     reg_e=0.1, reg_cl=0.1,verbose=True)
# ot_mapping_linear_crossed.fit(Xs=Xs, ys = ys, Xt=Xt, yt = yt)
# brown_pottok_transp_pot_crossed = ot_mapping_linear_crossed.transform(brown_pottok.reshape(-1,3))
   
   
# if (brown_pottok_transp_pot_crossed == brown_pottok_transp_crossed).all() : 
#     print ("POT and Pottok give same transformation - crossed")
# else : 
#     print ("ERROR : POT and Pottok do not give same transformation")

