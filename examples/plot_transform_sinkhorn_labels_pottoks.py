# -*- coding: utf-8 -*-
"""
===========================================================================
Pottok for image color adaptation with labels - OptimalTransportGridSearch
===========================================================================

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
# Optimal transport with SinkhornL1l2 with circular gridsearch
# --------------------------------------------------------------

gridsearch_transport_circular = pottok.OptimalTransportGridSearch(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
gridsearch_transport_circular.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt,scaler=False)
gridsearch_transport_circular.fit_circular()


# Best grid is {'reg_e': 1.0, 'reg_cl': 0.1}

##############################################################################
# Plot images
# -----------------------

brown_pottok_transp_circular = gridsearch_transport_circular.predict_transfer(brown_pottok.reshape(-1,3))
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
pl.imshow(brown_pottok_transp_circular.reshape(*brown_pottok.shape))
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()




##############################################################################
# Optimal transport with SinkhornL1l2 with crossed gridsearch
# --------------------------------------------------------------

gridsearch_transport_crossed = pottok.OptimalTransportGridSearch(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
gridsearch_transport_crossed.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt,scaler=False)
gridsearch_transport_crossed.fit_crossed()

# Best grid is {'reg_e': 0.1, 'reg_cl': 0.1}

##############################################################################
# Plot images
# -----------------------

brown_pottok_transp_crossed = gridsearch_transport_crossed.predict_transfer(brown_pottok.reshape(-1,3))
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
pl.imshow(brown_pottok_transp_crossed.reshape(*brown_pottok.shape))
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()


##############################################################################
# Comparison with pot
# ----------------------------------


ot_mapping_linear_circular = ot.da.SinkhornL1l2Transport(
    reg_e=1.0, reg_cl=0.1,verbose=True)
ot_mapping_linear_circular.fit(Xs=Xs, ys = ys, Xt=Xt, yt = yt)
brown_pottok_transp_pot_circular = ot_mapping_linear_circular.transform(brown_pottok.reshape(-1,3))

if (brown_pottok_transp_pot_circular == brown_pottok_transp_circular).all() : 
    print ("POT and Pottok give same transformation - circular")
else : 
    print ("ERROR : POT and Pottok do not give same transformation")
   
ot_mapping_linear_crossed = ot.da.SinkhornL1l2Transport(
    reg_e=0.1, reg_cl=0.1,verbose=True)
ot_mapping_linear_crossed.fit(Xs=Xs, ys = ys, Xt=Xt, yt = yt)
brown_pottok_transp_pot_crossed = ot_mapping_linear_crossed.transform(brown_pottok.reshape(-1,3))
   
   
if (brown_pottok_transp_pot_crossed == brown_pottok_transp_crossed).all() : 
    print ("POT and Pottok give same transformation - crossed")
else : 
    print ("ERROR : POT and Pottok do not give same transformation")

