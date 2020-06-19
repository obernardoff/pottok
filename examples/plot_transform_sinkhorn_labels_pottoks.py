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
# Load pottoks
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
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
gridsearch_transport.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt,scaler=False)
gridsearch_transport.fit_circular()
# gridsearch_transport.fit_crossed()


##############################################################################
# Plot images
# -----------------------

brown_pottok_transp = gridsearch_transport.predict_transfer(brown_pottok.reshape(-1,3))


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
pl.imshow(brown_pottok_transp.reshape(*brown_pottok.shape))
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()

###########



ot_mapping_linear = ot.da.SinkhornL1l2Transport(
    reg_e=1.0, reg_cl=0.1,verbose=True)
ot_mapping_linear.fit(Xs=Xs, ys = ys, Xt=Xt, yt = yt)


brown_pottok_transp_pot= ot_mapping_linear.transform(brown_pottok.reshape(-1,3))

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
pl.imshow(brown_pottok_transp_pot.reshape(*brown_pottok.shape))
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()

