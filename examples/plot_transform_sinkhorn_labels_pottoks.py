# -*- coding: utf-8 -*-
"""
=====================================================
OT for image color adaptation with labels
=====================================================

Using sinkhorn L1l2
"""

import numpy as np
import matplotlib.pylab as pl
import ot
import pottok


##############################################################################
# Generate data
# -------------

# Loading X and y
Xs,ys,Xt,yt = pottok.datasets.load_pottoks()

Xs = Xs/255
Xt = Xt/255

# Loading images array
brown_pottok,black_pottok = pottok.datasets.load_pottoks(return_X_y=False)
brown_pottok = brown_pottok/255
black_pottok = black_pottok/255

##############################################################################
# Optimal transport with SinkhornL1l2 with gridsearch
# ----------------------------------------------------

gridsearch_transport = pottok.OptimalTransportGridSearch(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e0,1e-1]))
gridsearch_transport.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt,scaler=False)
gridsearch_transport.fit_circular()


##############################################################################
# Plot images
# -----------------------

X1tl = gridsearch_transport.predict_transfer(brown_pottok.reshape(-1,3))
Image_mapping_gs = X1tl.reshape(*brown_pottok.shape)

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
pl.imshow(Image_mapping_gs)
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target)')

pl.show()
