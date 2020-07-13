#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:18:31 2020

@author: olivia
"""

import pottok
import ot
import museotoolbox as mtb
from sklearn.preprocessing import StandardScaler



#pour linux
# import warnings
# warnings.filterwarnings("ignore")
#################################

import os
if os.uname()[1] == 'kali':
    wdir = "/mnt/bigone/cours/Stage/2020/olivia/data/s2/"
else:
    wdir = "/home/olivia/Bureau/StageDynafor/donnees/decoupe/"
#image entière
source_image = wdir+'SITS_2019_crop_decoupe.tif'
target_image = wdir+'SITS_2017_crop_decoupe.tif' 
 


#le reste  
labels = wdir+'tree_species_references_decoupe.gpkg' #les labels
level1 = 'level1' # feuillu resineux
level2 = 'level2'
level3 = 'level3' # espece
group_label = 'rowid_BD'



########################################################
# ROT
# -----------------------------------------------------


print('---------------------------------------------------------')
print("ROT : régularisé entropie - Sinkhorn entropie")

rot = pottok.RasterOptimalTransport(transport_function=ot.da.SinkhornTransport,
                                        params=dict(reg_e=[1e-1,1e-0]),verbose = False)


rot.preprocessing(in_image_source=source_image,
                      in_image_target=target_image,
                      in_vector_source=labels,
                      in_vector_target=labels,
                      in_label_source=level3,
                      in_label_target=level3,
                      in_group_source=group_label,
                      in_group_target=group_label,
                      scaler=False)


# rot.fit_crossed()
rot.fit_circular()
rot_Xst = rot.predict_transfer(rot.Xs)

# yt_no_trans_rot, yt_transp_rot = rot.assess_transport(rot_Xst)
yt_no_trans_rot, yt_transp_rot = rot.assess_transport_circular(rot_Xst, ys = rot.ys, yt = rot.yt)

yt_rot = rot.yt

print('---------------------------------------------------------')
print('yt predit SANS transport', '\n', 'Nombre de True : ', sum(yt_rot == yt_no_trans_rot),"/",len(yt_rot))
print('---------------------------------------------------------')
print('yt predit AVEC transport','\n', 'Nombre de True : ',sum(yt_rot == yt_transp_rot),"/",len(yt_rot))
print('---------------------------------------------------------')

print('\n\n')


########################################################
# OTGS 
# -----------------------------------------------------


# Xs, ys, group_s = mtb.processing.extract_ROI(source_image, labels, level2, group_label)
# Xt, yt, group_t = mtb.processing.extract_ROI(target_image, labels, level2, group_label)



# print('---------------------------------------------------------')
# print("OTGS : régularisé entropie - Sinkhorn Transport")

# otgs = pottok.OptimalTransportGridSearch(transport_function=ot.da.SinkhornTransport,
#                                         params=dict(reg_e=[1e-1,1e-0]),verbose = False)

# otgs.preprocessing(Xs,ys=ys,Xt=Xt,yt=yt,scaler=False)


# otgs.fit_crossed()
# otgs_Xst = otgs.predict_transfer(Xs)

# yt_no_trans_otgs, yt_transp_otgs = otgs.assess_transport(otgs_Xst)

# yt_otgs = otgs.yt
# print('---------------------------------------------------------')
# print('yt predit SANS transport', '\n', 'Nombre de True : ', sum(yt_otgs == yt_no_trans_otgs),"/",len(yt_otgs))
# print('---------------------------------------------------------')
# print('yt predit AVEC transport', 'Nombre de True : ',sum(yt_otgs == yt_transp_otgs),"/",len(yt_otgs))
# print('---------------------------------------------------------')

























