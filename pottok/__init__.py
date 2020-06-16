# -*- coding: utf-8 -*-
# ================================================================
#              _   _        _    
#             | | | |      | |   
#  _ __   ___ | |_| |_ ___ | | __
# | '_ \ / _ \| __| __/ _ \| |/ /
# | |_) | (_) | |_| || (_) |   < 
# | .__/ \___/ \__|\__\___/|_|\_\
# | |                            
# |_|     
# ================================================================
# @author: Olivia Bernardoff, Nicolas Karasiak, Yousra Hamrouni & David Sheeren
# @git: https://github.com/obernardoff/pottok/
# ================================================================


# general libraries


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.preprocessing import StandardScaler #centrer-réduire
from itertools import product
import ot
import museotoolbox as mtb
import gdal


__version__ = "0.1-rc1"


class OptimalTransportGridSearch:
    """
    Initialize Python Optimal Transport suitable for validation.

    Parameters
    ----------
    transport_function: class  of ot.da, optional (default=ot.da.MappingTransport)
        from ot.da. e.g ot
    params_ot : dict
        parameters of the optimal transport funtion.
    verbose : boolean, optional (default=True)
        gives informations about the object
    """

    def __init__(self, 
                 transport_function = ot.da.MappingTransport, 
                 params=None, 
                 verbose=True):
        
        #stockage        
        self.transport_function = transport_function        
        self.params_ot = params       
        self.verbose = verbose    
        
        
    def preprocessing(self,
                     Xs,
                     ys=None,
                     Xt=None,
                     yt=None,
                     scaler = StandardScaler):
        """
        Stock the input parameters in the object and scaled it if it is asked.

        Parameters
        ----------
        Xs : array_like, shape (n_source_samples, n_features)
            Source domain array.
        ys : array_like, shape (n_source_samples,)
            Label source array (1d).
        Xt: array_like, shape (n_source_samples, n_features)
            Target domain array.
        yt: array_like, shape (n_source_samples,)
            Label target array (1d).
        scaler: scale function (default=StandardScaler)
            The function used to scale Xs and Xt
        """
     
        self._share_args(Xs=Xs,ys=ys,Xt=Xt,yt=yt,scaler=scaler)
        self._prefit(Xs,Xt)



    def fit_circular(self):
        """
        Learn domain adaptation model with circular tuning (fitting).
        
        Returns
        -------
        transport_model : object
            The output model fitted

        """
        
        if self._is_grid_search() :         
            self._find_best_parameters_circular(self.Xs, ys=self.ys, Xt=self.Xt, yt=self.yt)    
        else : 
            self.transport_model.fit(self.Xs, ys=self.ys, Xt=self.Xt, yt=self.yt) 
            
        return self.transport_model
    
    
        
    def fit_crossed(self,
                    group_s = None,
                    group_t = None,
                    cv_ai = StratifiedKFold(n_splits=2,shuffle=True,random_state=21),
                    cv_ot = StratifiedKFold(n_splits=2,shuffle=True,random_state=42),
                    classifier = RandomForestClassifier(),
                    parameters = dict(n_estimators=[100])):
        """
        Learn domain adaptation model with crossed tuning (fitting).
        
        Parameters
        -------
        group_s: array_like, shape (n_source_samples,)
            Polygon group of each label (1d)
        group_t: array_like, shape (n_source_samples,)
            Polygon group of each label (1d)
        cv_ai: cross-validation function
            cv used for the classifier learning.
            Allowed function from museotoolbox as scikit-learn.
        cv_ot: cross-validation function
            cv used for the train_test_split.
            Allowed function from museotoolbox as scikit-learn.
        classifier: training algorithm (default=RandomForestClassifier)
        
        Returns
        -------
        transport_model : object
            The output model fitted

        """

        self._share_args(group_s=group_s,group_t=group_t,cv_ot=cv_ot,cv_ai=cv_ai,classifier=classifier,parameters=parameters)
        
        
        Xt_valid, Xt_test, yt_valid, yt_test, groupt_valid, groupt_test = mtb.cross_validation.train_test_split(cv=cv_ai,X=self.Xt,y=self.yt,groups=self.group_t)                             
        self._share_args(Xt_valid=Xt_valid, Xt_test=Xt_test,yt_valid=yt_valid, yt_test=yt_test, groupt_valid=groupt_valid, groupt_test=groupt_test)
        
        #model with input parameters
        self._model = GridSearchCV(classifier,parameters,cv=cv_ai) 
        #save                   

        if self.params_ot is None:
            self.transport_model = self.transport_function()                        
        else:            
            # if grid search
            if self._is_grid_search():
                
                self._find_best_parameters_crossed(self.Xs,ys=self.ys,Xt=self.Xt,yt=self.yt,group_val=self.group_s)
                    
            else:
                # simply train with basic param
                self.transport_model.fit(self.Xs, ys=self.ys, Xt=self.Xt, yt=self.yt)
            
        return self.transport_model


    def predict_transfer(self, data):
        """
        Predict model using domain adaptation.

        Parameters
        ----------
        data : arr.
            Vector to transfer
        
        Return
        ----------        
        transport : arr
            tranfered vector 
        """
        
        if self.scaler is not False :
            data = self.Xs_scaler.transform(data)
                       
        data = self.transport_model.transform(data)   
        if self.scaler is not False :
            data = self.Xs_scaler.inverse_transform(data)
        return data
    

    def improve(self,Xs_transform, method = "test"):
        """
        OA comparison before and after OT

        Parameters
        ----------
        Xs_transform : array_like, shape (n_source_samples, n_features)
            Source domain array transformed after OT.
        method: str
            "test" or "all"
        """
        if method == "test" : 
            #avant transport
            self._model.fit(self.Xs,self.ys,self.group_s)
            y_pred_non_transport = self._model.predict(self.Xt_test)
            oa_non_transport = accuracy_score(self.yt_test,y_pred_non_transport)
            print("Avant transport, l'OA obtenu est de",oa_non_transport) 
            #apres transport       
            self._model.fit(Xs_transform,self.ys,self.group_s) 
            y_pred_transport = self._model.predict(self.Xt_test) 
            oa_transport = accuracy_score(self.yt_test,y_pred_transport) 
            print("Après transport, l'OA obtenu est de",oa_transport) 
            print("Il y a une amélioration de",round(oa_transport-oa_non_transport,4),"après transport")   
            
        else : 
        
            #avant transport
            self._model.fit(self.Xs,self.ys,self.group_s)
            y_pred_non_transport = self._model.predict(self.Xt)
            oa_non_transport = accuracy_score(self.yt,y_pred_non_transport)
            print("Avant transport, l'OA obtenu est de",oa_non_transport) 
            #apres transport       
            self._model.fit(Xs_transform,self.ys,self.group_s) 
            y_pred_transport = self._model.predict(self.Xt) 
            oa_transport = accuracy_score(self.yt,y_pred_transport)  
            print("Après transport, l'OA obtenu est de",oa_transport,"sur toute l'image")
            print("Il y a une amélioration de",round(oa_transport-oa_non_transport,4),"après transport (calcul sur toute l'image)")  
            
    
    
    
    def _share_args(self,**params):
        """
        Allow to stock each parameters enter by the user
        """
        for key,value in params.items():
            if key =='scaler':
                if value is False:
                    self._need_scale = False
                else:
                    self._need_scale = True
    
            self.__dict__[key] = value
            
    
    def _prefit(self,Xs,Xt) : 
        """
        Scale Xs and Xt

        Parameters
        ----------
        Xs : array_like, shape (n_source_samples, n_features)
            Source domain array.
        ys : array_like, shape (n_source_samples,)
            Label source array (1d).
        """
        
        if self.verbose:
            print('Learning Optimal Transport with ' +
                         str(self.transport_function.__name__) +
                         ' algorithm.')
            
            
        if self._need_scale:
            self.Xs_scaler = self._to_scale(Xs,self.scaler) #permet de stocker le scaler fitté
            self.Xs = self.Xs_scaler.transform(Xs)
            self.Xt_scaler = self._to_scale(Xt,self.scaler)
            self.Xt = self.Xt_scaler.transform(Xt)
            print("Xs and Xt are scaled")
        else : 
            print("Xs and Xt are not scaled")
        
        
    def _to_scale(self,data,method) :
        """
        Scale Xs and Xt

        Parameters
        ----------
        Xs : array_like, shape (n_source_samples, n_features)
            Source domain array.
        ys : array_like, shape (n_source_samples,)
            Label source array (1d).
            
        Return
        ----------
        scaler : fitted scaler on data
        """          
        scaler = method()
        scaler.fit(data) #pour vérifier qu'il y a quelque chose dedans après le .fit : scaler.scaler_
        return scaler

    
    def _is_grid_search(self):
        # search for gridSearch
        param_grid = []
        for key in self.params_ot.keys():
            if isinstance(self.params_ot.get(key), (list, np.ndarray)):
                param_grid.append(key)

        if param_grid == []:
            self.param_grid = False
        else:
            self.param_grid = param_grid
            self.params_ot = self.params_ot.copy()

        if self.param_grid:
            return True
        else:
            return False

    def _generate_params_from_grid_search(self):
        
        hyper_param = {key: self.params_ot[key] for key in self.param_grid} 
        items = sorted(hyper_param.items())
        keys, values = zip(*items)
        for v in product(*values):
            params_to_add = dict(zip(keys, v))
            self.params_ot.update(params_to_add)
            yield self.params_ot



    def _find_best_parameters_crossed(self,Xs,ys,Xt,yt,group_val):
        """
        Find the best parameters of the transport function with crossed method
        
        Parameters
        ----------
        Xs : array_like, shape (n_source_samples, n_features)
            Source domain array.
        ys : array_like, shape (n_source_samples,)
            Label source array (1d).
        Xt: array_like, shape (n_source_samples, n_features)
            Target domain array.
        yt: array_like, shape (n_source_samples,)
            Label target array (1d). 
        group_val : array_like, shape (n_source_samples,)
            Polygon group of each label (1d)
        """     
        self.best_score = None
        #boucle qui test chaque hyperparametres
        for gridOT in self._generate_params_from_grid_search():
            print(gridOT)
            transport_model_tmp = self.transport_function(**gridOT)#modele de transport pour chaque combinaison de parametres
            #transport
            transport_model_tmp.fit(Xs=Xs,ys=ys, Xt=Xt,yt=yt)
            Xs_transform = transport_model_tmp.transform(Xs=Xs)#transformation des Xs 
            #apprentissage du nouveau modele sur Xs_transform
            self._model.fit(Xs_transform,ys,group_val)
            print('Accord global issu de la validation croisée : '+str(self._model.best_score_))
            print('Le meilleur paramètre est : '+str(self._model.best_params_))
            #prediction sur les Xt_valid           
            yt_pred_valid = self._model.predict(self.Xt_valid)
            oa_transport = accuracy_score(self.yt_valid, yt_pred_valid)
            print("Après transport, l'OA obtenu est de",oa_transport ) 
            print("-------------------------------------------------")
            #meilleurs parametres
            if self.best_score is None or oa_transport > self.best_score:
                self.best_score = oa_transport
                self.best_params = gridOT.copy() #stocke les meilleurs parametres       
                self.transport_model = transport_model_tmp #stocke le modele fitté av les meilleurs parametres = gagne du temps car pas besoin de le refaire à la fin du fit_crossed
        if self.verbose:
            print('Best grid is ' +
                         str(self.best_params))
            print('Best score is ' +
                         str(self.best_score))
        

    def _find_best_parameters_circular(self, Xs, ys, Xt, yt):
        """
        Find the best parameters of the transport function with circular method
        
        Parameters
        ----------
        Xs : array_like, shape (n_source_samples, n_features)
            Source domain array.
        ys : array_like, shape (n_source_samples,)
            Label source array (1d).
        Xt: array_like, shape (n_source_samples, n_features)
            Target domain array.
        yt: array_like, shape (n_source_samples,)
            Label target array (1d). 
        """  
        self.best_score = None
        for gridOT in self._generate_params_from_grid_search():
            transport_model_tmp = self.transport_function(**gridOT)
            transport_model_tmp.fit(Xs, ys, Xt, yt)
            from ot.da import BaseTransport
            transp_Xt = BaseTransport.inverse_transform(#pour faire le retour
                transport_model_tmp, Xs=Xs, ys=ys, Xt=Xt, yt=yt)
            currentScore = mean_squared_error(Xs, transp_Xt) #regarde les différences d'allé retour

            if self.verbose:
                print(
                    'RMSE is : ' + str(currentScore))

            if self.best_score is None or self.best_score > currentScore:
                self.best_score = currentScore
                self.best_params = gridOT.copy() #met à jour les meilleurs paramètres si on a obtenu le 
                                                 #meilleur score
                self.transport_model = transport_model_tmp
        if self.verbose:
            print('Best grid is ' +
                         str(self.best_params))
            print('Best score is ' +
                         str(self.best_score))

        














class RasterOptimalTransport(OptimalTransportGridSearch):

    def __init__(self, 
                 transport_function = ot.da.MappingTransport, 
                 params=None, 
                 verbose=True):
        """
        Initialize Python Optimal Transport for raster processing.
        """             
        super().__init__(transport_function,params,verbose)
        

     

    def preprocessing(self,                  
                     in_image_source = None,
                     in_image_target = None,
                     in_vector_source = None,
                     in_vector_target = None,
                     in_label_source = None,
                     in_label_target = None,
                     in_group_source = None,
                     in_group_target = None,
                     scaler = StandardScaler):
        """
        Scale all image (if it is asked) and stock the input parameters in the object .
        
        Parameters
        ----------
        in_image_source : str.
            source image (*.tif) --> path + file name 
        in_image_target : str.
            target image (*.tif) --> path + file name 
        in_vector_source : str.
            labels (*.gpkg) --> path + file name 
        in_vector_target : str.
            labels (*.gpkg) --> path + file name 
        in_label_source : str.
            name of the label colum in vector file (source)
        in_label_source : str.
            name of the label colum in vector file (target)
        in_group_source : str.
            name of the group colum of each polygon in vector file (source)
        in_group_target : str.
            name of the group colum of each polygon in vector file (target)
        scaler: scale function (default=StandardScaler)
            The function used to scale source and target image
        """
       
        self._share_args(in_image_source=in_image_source,
                         in_image_target=in_image_target,
                         in_vector_source=in_vector_source,
                         in_vector_target=in_vector_target,
                         in_label_source=in_label_source,
                         in_label_target=in_label_target,
                         in_group_source=in_group_source,
                         in_group_target=in_group_target,
                         scaler=scaler)        


        #faire extract ROI à partir de l'image pour obtenir les positions
        Xs,ys,group_s,pos_s = mtb.processing.extract_ROI(self.in_image_source,
                                                   self.in_vector_source,
                                                   self.in_label_source,
                                                   self.in_group_source,                                                                                      
                                                   get_pixel_position=True) # Xsource ysource
        
        pos_s = pos_s.astype(int) #pour les mettre en entier
        
        Xt,yt,group_t,pos_t = mtb.processing.extract_ROI(self.in_image_target,
                                                   self.in_vector_target,
                                                   self.in_label_target,
                                                   self.in_group_target,
                                                   get_pixel_position=True) # Xtarget ytarget
        pos_t = pos_t.astype(int) #pour les mettre en entier        
        
        
        
        #1.Scale de l'image 
        if self._need_scale :
                               
            #source
            source = gdal.Open(self.in_image_source)
            bande_scale_source = np.zeros((source.RasterYSize,source.RasterXSize,source.RasterCount))#pour stocker toutes les bandes --> shape : lignes, colonnes, bandes
            Xs_scale = np.zeros((Xs.shape[0],Xs.shape[1]))#pour stocker les Xs  --> shape : pixels, bandes
            
            for bande in range(source.RasterCount) :
                bande_i = source.GetRasterBand(bande+1) 
                source_bande_i=bande_i.ReadAsArray()   
                scale_bande = self._to_scale(source_bande_i,self.scaler)  
                for i in range(Xs.shape[0]):
                    Xs_scale[i,bande] = scale_bande[pos_s[i,1],pos_s[i,0]]   
                bande_scale_source[:,:,bande] = scale_bande 


            bande_scale_source_reshape = self.im2mat(bande_scale_source)


            #target
            target = gdal.Open(self.in_image_target)
            bande_scale_target = np.zeros((target.RasterYSize,target.RasterXSize,target.RasterCount))#idem
            Xt_scale = np.zeros((Xt.shape[0],Xt.shape[1]))#idem
            
            for bande in range(target.RasterCount) :
                bande_i = target.GetRasterBand(bande+1) 
                target_bande_i=bande_i.ReadAsArray()   
                scale_bande = self._to_scale(target_bande_i,self.scaler)  
                for i in range(Xt.shape[0]):
                    Xt_scale[i,bande] = scale_bande[pos_t[i,1],pos_t[i,0]]   
                bande_scale_target[:,:,bande] = scale_bande 
            
            
            bande_scale_target_reshape = self.im2mat(bande_scale_target)
            
            
            #c. stockage des données extraites
            self.Xs_non_scale = Xs
            self.Xs = Xs_scale
            self.ys = ys
            self.Xt_non_scale = Xt
            self.Xt = Xt_scale
            self.yt = yt
            self.group_s = group_s
            self.group_t = group_t
            self.image_scale_source = bande_scale_source
            self.image_scale_target = bande_scale_target
            self.image_scale_source_reshape = bande_scale_source_reshape
            self.image_scale_target_reshape = bande_scale_target_reshape
            print ("Image is scaled")
            
        #2.Si pas de scale, on extrait juste les Xs.
        else : 

            #source
            source = gdal.Open(self.in_image_source)
            bande_source = np.zeros((source.RasterYSize,source.RasterXSize,source.RasterCount))#pour stocker toutes les bandes --> shape : lignes, colonnes, bandes            
            for bande in range(source.RasterCount) :
                bande_i = source.GetRasterBand(bande+1) 
                source_bande_i=bande_i.ReadAsArray()   
                bande_source[:,:,bande] = source_bande_i 


            bande_source_reshape = self.im2mat(bande_source)



            #target
            target = gdal.Open(self.in_image_target)
            bande_target = np.zeros((target.RasterYSize,target.RasterXSize,target.RasterCount))#idem
            Xt_scale = np.zeros((Xt.shape[0],Xt.shape[1]))#idem
            
            for bande in range(target.RasterCount) :
                bande_i = target.GetRasterBand(bande+1) 
                target_bande_i=bande_i.ReadAsArray()    
                bande_target[:,:,bande] = target_bande_i             
            
            
            bande_target_reshape = self.im2mat(bande_target)
            
            #c.stockage des données extraites
            self.Xs = Xs
            self.ys = ys
            self.Xt = Xt
            self.yt = yt
            self.group_s = group_s
            self.group_t = group_t            
            self.image_source = bande_source
            self.image_target = bande_target        
            self.image_source_reshape = bande_source_reshape
            self.image_target_reshape = bande_target_reshape   
            print ("Image is not scaled")
    
    
    def predict_transfer(self, data):
        """
        Predict model using domain adaptation.

        Parameters
        ----------        
        data : arr.
            Vector to transfer
        
        Return
        ----------        
        transport : arr
            tranfered vector 
        """                       
        transport = self.transport_model.transform(data)   
        return transport

    #fonctions rajoutées pour avoir les mêmes dimensions 
    def im2mat(self,img):
        """
        Converts an image to matrix (one pixel per line)
        
        Parameters
        ----------
        img : arr.
            array to reshape
        
        Return
        ----------        
        reshape img             
        """
        return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    
    
    def mat2im(self,X, shape):
        """
        Converts back a matrix to an image
        
        Parameters
        ----------
        X : arr.
            array to reshape
        shape
            shape of the image 
        
        Return
        ----------        
        reshape img                     
        """
        return X.reshape(shape)    


    def _to_scale(self,data,method) : 
        """
        Parameters
        ----------
        data : arr.
            data to scale
        method : function.
            scaler method
        
        Returns
        -------
        scale : arr.
            scaled data
        """        
        
        scaler = method()
        scaler.fit(data)
        scale = scaler.transform(data)
        return scale