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
from sklearn.preprocessing import StandardScaler  # centrer-réduire



source_image,source_vector,target_image,target_vector = pottok.datasets.load_pottoks(return_only_path = True)

brown_pottok,black_pottok = pottok.datasets.load_pottoks(return_X_y=False)
brown_pottok = brown_pottok/255
black_pottok = black_pottok/255


label = 'level' 



##############################################################################
# Optimal transport with SinkhornL1l2 with circular gridsearch
# --------------------------------------------------------------

raster_transport_circular = pottok.RasterOptimalTransport(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
raster_transport_circular.preprocessing(in_image_source = source_image,
                   in_image_target = target_image,
                   in_vector_source = source_vector,
                   in_vector_target = target_vector,
                   in_label_source = label,
                   in_label_target = label,
                   scaler = StandardScaler)


raster_transport_circular.fit_circular()


# Best grid is {'reg_e': 1.0, 'reg_cl': 1.0}

#############################################################################
# Plot images
# -----------------------

brown_pottok_transp_circular = raster_transport_circular.predict_transfer(raster_transport_circular.source.reshape(-1,3))

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
pl.imshow(brown_pottok_transp_circular.reshape(*brown_pottok.shape)/255)
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()




# ##############################################################################
# # Optimal transport with SinkhornL1l2 with crossed gridsearch
# # --------------------------------------------------------------

raster_transport_crossed = pottok.RasterOptimalTransport(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
raster_transport_crossed.preprocessing(in_image_source = source_image,
                   in_image_target = target_image,
                   in_vector_source = source_vector,
                   in_vector_target = target_vector,
                   in_label_source = label,
                   in_label_target = label,
                   scaler = StandardScaler)


raster_transport_crossed.fit_crossed()


# Best grid is {'reg_e': 0.1, 'reg_cl': 0.1}

#############################################################################
# Plot images
# -----------------------

brown_pottok_transp_crossed = raster_transport_crossed.predict_transfer(raster_transport_crossed.source.reshape(-1,3))

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
pl.imshow(brown_pottok_transp_crossed.reshape(*brown_pottok.shape)/255)
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()





