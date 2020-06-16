# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:34:33 2020

@author: Olivia
"""


from pottok import OptimalTransportGridSearch,RasterOptimalTransport
import museotoolbox as mtb
import ot
import museopheno as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler #centrer-réduire

    

user = 'olivia'
if user == 'nicolas':
    wdir = "/mnt/bigone/cours/Stage/2020/olivia/data/s2/"
    use_decoupe = False
elif user == 'olivia':
    wdir = "C:/Users/Olivia/Documents/StageDynafor/donnees/decoupe/"
    use_decoupe = True

if use_decoupe is True:
    decoupe_str = '_decoupe'
else:
    decoupe_str = ''



source_image = 'SITS_2019_crop_decoupe.tif'
target_image = 'SITS_2017_crop_decoupe.tif'
vector = 'tree_species_references_decoupe.gpkg' #le même pour sources et target
label = 'level3' #le même pour sources et target
group = 'rowid_BD' #le même pour sources et target


Xs,ys,group_s = mtb.processing.extract_ROI(wdir+source_image,wdir+vector,label,group) 
Xt,yt,group_t = mtb.processing.extract_ROI(wdir+target_image,wdir+vector,label,group) 

s2_ts = mp.sensors.Sentinel2() #calcul des ndvi
Xs_ndvi = s2_ts.generate_index(Xs, s2_ts.get_index_expression('ACORVI'))  #calcul des ndvi
Xt_ndvi = s2_ts.generate_index(Xs, s2_ts.get_index_expression('ACORVI'))  #calcul des ndvi

###################
#Xs et Xt normaux
###################

#crossed non scale

print("Xs et Xt non scaled - crossed")
test_crossed = OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0,0.1], eta=[10.0,2.0]))

test_crossed.preprocessing(Xs,ys,Xt,yt,scaler = False)
#print(test_crossed.Xs)
#print(test_crossed.Xt)

test_crossed.fit_crossed(group_t=group_t,classifier=RandomForestClassifier(random_state=42))
Xs_transform_crossed = test_crossed.predict_transfer(Xs)
test_crossed.improve(Xs_transform_crossed)
print("----------------------------------")
print("----------------------------------")

#crossed scale

print("Xs et Xt scaled - crossed")
test_crossed_scale = OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0,0.1], eta=[10.0,2.0]))

test_crossed_scale.preprocessing(Xs,ys,Xt,yt,scaler = StandardScaler)
#print(test_crossed_scale.Xs)
#print(test_crossed_scale.Xt)

test_crossed_scale.fit_crossed(group_t=group_t,classifier=RandomForestClassifier(random_state=42))
Xs_transform_crossed = test_crossed_scale.predict_transfer(Xs)
test_crossed_scale.improve(Xs_transform_crossed)
print("----------------------------------")
print("----------------------------------")


#circular

print("Xs et Xt scaled - circular")
test_circular = OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0,0.1], eta=[10.0,2.0]))
test_circular.preprocessing(Xs,ys,Xt,yt,scaler = StandardScaler)
#print(test_circular.Xs)
#print(test_circular.Xt)


test_circular.fit_circular()
print("----------------------------------")
print("----------------------------------")



###################
#Xs et Xt NDVI
###################

print("Xs et Xt scaled NDVI - crossed")
test_ndvi = OptimalTransportGridSearch(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0,0.1], eta=[10.0,2.0]))
test_ndvi.preprocessing(Xs,ys,Xt,yt,scaler = StandardScaler)
#print(test_ndvi.Xs)
#print(test_ndvi.Xt)

test_ndvi.fit_crossed(group_t=group_t,classifier=RandomForestClassifier(random_state=42))
Xs_transform_crossed = test_ndvi.predict_transfer(Xs)
test_ndvi.improve(Xs_transform_crossed)
print("----------------------------------")
print("----------------------------------")


###################
#avec toute l'image 
###################

print("Xs et Xt scaled on all image - crossed")
test_raster = RasterOptimalTransport(transport_function = ot.da.MappingTransport,
                              params=dict(mu=[2.0,0.1], eta=[10.0,2.0]))

test_raster.preprocessing(in_image_source = wdir+source_image,
                     in_image_target = wdir+target_image,
                     in_vector_source = wdir+vector,
                     in_vector_target = wdir+vector,
                     in_label_source = label,
                     in_label_target = label,
                     in_group_source = group,
                     in_group_target = group,
                     scaler = StandardScaler)

image_scale_source_reshape = test_raster.image_scale_source_reshape

test_raster.fit_crossed(group_t=group_t,classifier=RandomForestClassifier(random_state=42))

#Xs transport 
Xs_transform_crossed = test_raster.predict_transfer(test_raster.Xs)
test_raster.improve(Xs_transform_crossed)

#Raster transport 
raster_s_transform_crossed = test_raster.predict_transfer(image_scale_source_reshape)







