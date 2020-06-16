# -*- coding: utf-8 -*-
"""
=====================================================
OT for image color adaptation with mapping estimation
=====================================================

OT for domain adaptation with image color adaptation [6] with mapping
estimation [8].

[6] Ferradans, S., Papadakis, N., Peyre, G., & Aujol, J. F. (2014). Regularized
discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

[8] M. Perrot, N. Courty, R. Flamary, A. Habrard, "Mapping estimation for
discrete optimal transport", Neural Information Processing Systems (NIPS), 2016.

"""

# Authors: Remi Flamary <remi.flamary@unice.fr>
#          Stanislas Chambon <stan.chambon@gmail.com>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pylab as pl
import ot
import pottok
r = np.random.RandomState(42)


def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img,amin=0,amax=1):
    return np.clip(img, amin, amax)


##############################################################################
# Generate data
# -------------

# Loading images
I1,I2 = pottok.datasets.load_pottoks()


X1 = im2mat(I1).astype(np.int16)
X2 = im2mat(I2).astype(np.int16)

# training samples
nb = 1000
idx1 = r.randint(X1.shape[0], size=(nb,))
idx2 = r.randint(X2.shape[0], size=(nb,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]


##############################################################################
# Domain adaptation for pixel distribution transfer
# -------------------------------------------------

gridsearch_transport = pottok.OptimalTransportGridSearch(transport_function=ot.da.SinkhornTransport,
                                        params=dict(reg_e=[1,1e5],max_iter=[0.01,0.1]))
gridsearch_transport.preprocessing(Xs=Xs,Xt=Xt,scaler=False)
gridsearch_transport.fit_circular()


##############################################################################
# Plot original images
# --------------------

pl.figure(1, figsize=(6.4, 3))
pl.subplot(1, 2, 1)
pl.imshow(I1)
pl.axis('off')
pl.title('Image 1')

pl.subplot(1, 2, 2)
pl.imshow(I2)
pl.axis('off')
pl.title('Image 2')
pl.tight_layout()




##############################################################################
# Plot transformed images
# -----------------------


X1tl = gridsearch_transport.predict_transfer(X1)
Image_mapping_gs = mat2im(X1tl, I1.shape)

Image_mapping_gs  = minmax(Image_mapping_gs,0,255).astype(np.int16)

pl.figure(2, figsize=(10,8))

pl.subplot(2, 2, 1)
pl.imshow(I1)
pl.axis('off')
pl.title('Im. 1')

pl.subplot(2, 2, 3)
pl.imshow(I2)
pl.axis('off')
pl.title('Im. 2')

pl.subplot(2, 2, 4)
pl.imshow(Image_mapping_gs)
pl.axis('off')
pl.title('MappingTransport (linear)')

pl.show()
