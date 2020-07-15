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
from sklearn.preprocessing import StandardScaler,MinMaxScaler  # centrer-réduire



source_image,source_vector,target_image,target_vector = pottok.datasets.load_pottoks(return_only_path = True)

brown_pottok,black_pottok = pottok.datasets.load_pottoks(return_X_y=False)
brown_pottok = brown_pottok/255
black_pottok = black_pottok/255


label = 'level' 
    
##############################################################################
# Optimal transport with SinkhornL1l2 with circular gridsearch
# --------------------------------------------------------------

raster_transport_circular = pottok.RasterOptimalTransport(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1]))
raster_transport_circular.preprocessing(image_source = source_image,
                   image_target = target_image,
                   vector_source = source_vector,
                   vector_target = target_vector,
                   label_source = label,
                   label_target = label,
                   scaler = MinMaxScaler)


raster_transport_circular.fit_circular()


# Best grid is {'reg_e': 1.0, 'reg_cl': 1.0}

#############################################################################
# Plot images
# -----------------------

Xt_transp_unscaled,  Xt_transp_scaled = raster_transport_circular.predict_transfer(raster_transport_circular.source)

pl.figure(1, figsize=(10,8))

pl.subplot(2, 2, 1)
pl.imshow(brown_pottok)
pl.axis('off')
pl.title('Brown pottok (Source)')

pl.subplot(2, 2, 3)
pl.imshow(black_pottok)
pl.axis('off')
pl.title('Black pottok (Target)')

pl.subplot(2, 2, 4)
pl.imshow(Xt_transp_unscaled.reshape(*brown_pottok.shape)/255)
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()




# ##############################################################################
# # Optimal transport with SinkhornL1l2 with crossed gridsearch
# # --------------------------------------------------------------

raster_transport_crossed = pottok.RasterOptimalTransport(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1]))
raster_transport_crossed.preprocessing(image_source = source_image,
                   image_target = target_image,
                   vector_source = source_vector,
                   vector_target = target_vector,
                   label_source = label,
                   label_target = label,
                   scaler = MinMaxScaler)


raster_transport_crossed.fit_crossed()


# Best grid is {'reg_e': 0.1, 'reg_cl': 0.1}

#############################################################################
# Plot images
# -----------------------

Xt_transp_unscaled,  Xt_transp_scaled = raster_transport_crossed.predict_transfer(raster_transport_crossed.source)

pl.figure(2, figsize=(10,8))

pl.subplot(2, 2, 1)
pl.imshow(brown_pottok)
pl.axis('off')
pl.title('Brown pottok (Source)')

pl.subplot(2, 2, 3)
pl.imshow(black_pottok)
pl.axis('off')
pl.title('Black pottok (Target)')

pl.subplot(2, 2, 4)
pl.imshow(Xt_transp_scaled.reshape(*brown_pottok.shape))
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()





