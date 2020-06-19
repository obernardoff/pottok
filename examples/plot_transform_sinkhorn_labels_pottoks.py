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


def im2mat(img):
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def mat2im(X, shape):
    return X.reshape(shape)    

def minmax(img):
    return np.clip(img, 0, 1)


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

brown_pottok_reshape = im2mat(brown_pottok)
black_pottok_reshape = im2mat(black_pottok)

##############################################################################
# Optimal transport with SinkhornL1l2 with gridsearch
# ----------------------------------------------------

gridsearch_transport = pottok.OptimalTransportGridSearch(transport_function=ot.da.SinkhornL1l2Transport,
                                        params=dict(reg_e=[1e-1,1e-0], reg_cl=[1e-1,1e-0]))
gridsearch_transport.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt,scaler=False)
#gridsearch_transport.fit_circular()
gridsearch_transport.fit_crossed()


##############################################################################
# Plot images
# -----------------------

X1tl = gridsearch_transport.predict_transfer(brown_pottok_reshape)
Image_mapping_gs = minmax(mat2im(X1tl,brown_pottok.shape))


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
pl.title('SinkhornL1l2 (Source to Target with labels)')

pl.show()


##############################################################################
# avec POT: 
# ----------------------- 


ot_mapping_linear = ot.da.MappingTransport(
    mu=2.0, eta=10.0, bias=True, max_iter=20, verbose=True)
ot_mapping_linear.fit(Xs=Xs, Xt=Xt)


X1tl_pot = ot_mapping_linear.transform(Xs=brown_pottok_reshape)
Image_mapping_gs_pot = minmax(mat2im(X1tl_pot,brown_pottok.shape))


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
pl.imshow(Image_mapping_gs_pot)
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target)')

pl.show()




##############################################################################
# WAY 2
# -----------------------

##############################################################################
#Generate data 
# -----------------------


brownpottok = pl.imread('C:/Users/Olivia/Documents/StageDynafor/recherche/clone_git/pottok/pottok/datasets/brownpottok.jpg').astype(np.float64) / 256
blackpottok = pl.imread('C:/Users/Olivia/Documents/StageDynafor/recherche/clone_git/pottok/pottok/datasets/blackpottok.jpg').astype(np.float64) / 256


X1_brown = im2mat(brownpottok)
X2_black = im2mat(blackpottok)

# training samples
nb = 1000

r = np.random.RandomState(42)

idx1 = r.randint(X1_brown.shape[0], size=(nb,))
idx2 = r.randint(X2_black.shape[0], size=(nb,))

Xs = X1_brown[idx1, :]
Xt = X2_black[idx2, :]


##############################################################################
# Optimal transport with MappingTransport with gridsearch
# ----------------------------------------------------------

gridsearch_transport = pottok.OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0], eta=[10.0], bias=True,
                              max_iter=20, verbose=True))
gridsearch_transport.preprocessing(Xs=Xs,ys=None,Xt=Xt,yt=None,scaler=False)
gridsearch_transport.fit_circular()



##############################################################################
# Plot images - way2
# -----------------------


X1tl = gridsearch_transport.predict_transfer(X1_brown)
Image_mapping_gs = minmax(mat2im(X1tl,brownpottok.shape))

pl.figure(3, figsize=(10,8))

pl.subplot(3, 2, 1)
pl.imshow(brownpottok)
pl.axis('off')
pl.title('Brown pottok (Source)')

pl.subplot(3, 2, 3)
pl.imshow(blackpottok)
pl.axis('off')
pl.title('Black pottok (Target)')

pl.subplot(3, 2, 4)
pl.imshow(Image_mapping_gs)
pl.axis('off')
pl.title('MappingTransport (Source to Target)')

pl.show()


##############################################################################
# avec POT: 
# ----------------------- 


ot_mapping_linear = ot.da.MappingTransport(
    mu=2.0, eta=10.0, bias=True, max_iter=20, verbose=True)
ot_mapping_linear.fit(Xs=Xs, Xt=Xt)


X1tl = ot_mapping_linear.transform(Xs=X1_brown)
Image_mapping_gs = minmax(mat2im(X1tl,brownpottok.shape))


pl.figure(2, figsize=(10,8))

pl.subplot(2, 2, 1)
pl.imshow(brownpottok)
pl.axis('off')
pl.title('Brown pottok (Source)')

pl.subplot(2, 2, 3)
pl.imshow(blackpottok)
pl.axis('off')
pl.title('Black pottok (Target)')

pl.subplot(2, 2, 4)
pl.imshow(Image_mapping_gs)
pl.axis('off')
pl.title('SinkhornL1l2 (Source to Target)')

pl.show()