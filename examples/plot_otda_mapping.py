# -*- coding: utf-8 -*-
"""
===========================================
OT mapping estimation for domain adaptation
===========================================

This example presents how to use MappingTransport to estimate at the same
time both the coupling transport and approximate the transport map with either
a linear or a kernelized mapping as introduced in [8].

[8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
"Mapping estimation for discrete optimal transport",
Neural Information Processing Systems (NIPS), 2016.
"""

# Authors: Remi Flamary <remi.flamary@unice.fr>
#          Stanislas Chambon <stan.chambon@gmail.com>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib.pylab as pl
import ot
import pottok

##############################################################################
# Generate data
# -------------

n_source_samples = 100
n_target_samples = 100
theta = 2 * np.pi / 20
noise_level = 0.1

Xs, ys = ot.datasets.make_data_classif(
    'gaussrot', n_source_samples, nz=noise_level)
Xs_new, _ = ot.datasets.make_data_classif(
    'gaussrot', n_source_samples, nz=noise_level)
Xt, yt = ot.datasets.make_data_classif(
    'gaussrot', n_target_samples, theta=theta, nz=noise_level)

# one of the target mode changes its variance (no linear mapping)
Xt[yt == 2] *= 3
Xt = Xt + 4

##############################################################################
# Plot data
# ---------

pl.figure(1, (10, 5))
pl.clf()
pl.scatter(Xs[:, 0], Xs[:, 1], c=ys, marker='+', label='Source samples')
pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o', label='Target samples')
pl.legend(loc=0)
pl.title('Source and target distributions')


##############################################################################
# Instantiate the different transport algorithms and fit them
# -----------------------------------------------------------

#######################################################
# MappingTransport with pottok
# -----------------------------------



# MappingTransport with circular validation
pottok_circular = pottok.OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0,0.1], eta=[10.0,2.0], kernel=["gaussian","linear"], bias=True,
                              max_iter=20, verbose=True))
pottok_circular.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt)

pottok_circular.fit_circular()

transp_Xs_linear_circular = pottok_circular.predict_transfer(Xs)

transp_Xs_linear_new_circular = pottok_circular.predict_transfer(Xs_new)


# MappingTransport with crossed validation

pottok_crossed = pottok.OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0,0.1], eta=[10.0,2.0], kernel=["gaussian","linear"], bias=True,
                              max_iter=20, verbose=True))

pottok_crossed.preprocessing(Xs=Xs,ys=ys,Xt=Xt,yt=yt)

pottok_crossed.fit_crossed()

transp_Xs_linear_crossed = pottok_circular.predict_transfer(Xs)

transp_Xs_linear_new_crossed = pottok_circular.predict_transfer(Xs_new)


###############################################################################
## Plot transported samples
## ------------------------
#
#pl.figure(2)
#pl.clf()
#pl.subplot(2, 2, 1)
#pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
#           label='Target samples', alpha=.2)
#pl.scatter(transp_Xs_linear[:, 0], transp_Xs_linear[:, 1], c=ys, marker='+',
#           label='Mapped source samples')
#pl.title("Bary. mapping (linear)")
#pl.legend(loc=0)
#
#pl.subplot(2, 2, 2)
#pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
#           label='Target samples', alpha=.2)
#pl.scatter(transp_Xs_linear_new[:, 0], transp_Xs_linear_new[:, 1],
#           c=ys, marker='+', label='Learned mapping')
#pl.title("Estim. mapping (linear)")
#
#pl.subplot(2, 2, 3)
#pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
#           label='Target samples', alpha=.2)
#pl.scatter(transp_Xs_gaussian[:, 0], transp_Xs_gaussian[:, 1], c=ys,
#           marker='+', label='barycentric mapping')
#pl.title("Bary. mapping (kernel)")
#
#pl.subplot(2, 2, 4)
#pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
#           label='Target samples', alpha=.2)
#pl.scatter(transp_Xs_gaussian_new[:, 0], transp_Xs_gaussian_new[:, 1], c=ys,
#           marker='+', label='Learned mapping')
#pl.title("Estim. mapping (kernel)")
#pl.tight_layout()
#
#pl.show()
