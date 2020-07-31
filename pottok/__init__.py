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
"""
The :mod:`pottok` module gathers available classes and function for `pottok`.
"""

from . import datasets

# general libraries


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler  # centrer-réduire
from itertools import product
import ot
import museotoolbox as mtb
import gdal
import pickle

__version__ = "0.1-rc2"


class OptimalTransportGridSearch:
    """
    Initialize Python Optimal Transport suitable for validation.

    Parameters
    ----------
    transport_function: class of ot.da, optional (default=ot.da.MappingTransport)
        from ot.da. e.g ot
    params_ot : dict, optional (default=None)
        parameters of the optimal transport funtion.
    verbose : boolean, optional (default=True)
        Gives informations about the object
    """

    def __init__(self,
                 transport_function=ot.da.MappingTransport,
                 params=None,
                 verbose=True):

        # stockage
        self.transport_function = transport_function
        self.params_ot = params
        self.verbose = verbose

    def _convert_to_float64(self,array):
        """
        If array is integer, convert to float64.
        In any case, return array !
        """
        
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float64)
        
        return array
        
    def preprocessing(self,
                      Xs,
                      ys=None,
                      Xt=None,
                      yt=None,
                      group_s=None,
                      group_t=None,
                      scaler=False):
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
        scaler: scale function (default=False)
            The function used to scale Xs and Xt
        """

        self._share_args(Xs=self._convert_to_float64(Xs), ys=ys, Xt=self._convert_to_float64(Xt), yt=yt, group_s=group_s, group_t=group_t, scaler=scaler)
        self._prefit(Xs, Xt)

    def fit_circular(self,
                     metrics=mean_squared_error,
                     greater_is_better=False):
        """
        Learn domain adaptation model with circular tuning (fitting).

        Parameters
        -----------
        metrics : function, optional (default=mean_squared_error)
            Need a function that takes two array as parameters
        greater_is_better : bool, optional (default=False)
            If mean_squared_error, the lower is the better. Else, if overall accuracy fo rexample greater_is_better is True.

        Returns
        --------
        transport_model : object
            The output model fitted

        """
        self.metrics = metrics
        self.greater_is_better = greater_is_better
        
        print(self.Xs)
        if self._is_grid_search():
            self._find_best_parameters_circular(
                self.Xs, ys=self.ys, Xt=self.Xt, yt=self.yt)
        else:
            self.transport_model = self.transport_function(**self.params_ot)
            self.transport_model.fit(
                self.Xs, ys=self.ys, Xt=self.Xt, yt=self.yt)

        return self.transport_model

    def fit_crossed(self,
                    cv_ai=StratifiedKFold(n_splits=2, shuffle=True, random_state=21),
                    cv_ot=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    classifier=RandomForestClassifier(random_state=42),
                    parameters=dict(n_estimators=[100]),
                    yt_use=True,
                    Xt_valid=None,
                    Xt_test=None,
                    yt_valid=None,
                    yt_test=None):
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
            cv used for the tratest_split.
            Allowed function from museotoolbox as scikit-learn.
        classifier: training algorithm (default=RandomForestClassifier)

        Returns
        -------
        transport_model : object
            The output model fitted

        """

        self._share_args(
            cv_ot=cv_ot,
            cv_ai=cv_ai,
            classifier=classifier,
            parameters=parameters,
            yt_use=yt_use)
        
        
        if Xt_valid is not None : 
            self._share_args(
            Xt_valid=Xt_valid,
            Xt_test=Xt_test,
            yt_valid=yt_valid,
            yt_test=yt_test)
            
        else : 

            if self.group_s is None:
    
                Xt_valid, Xt_test, yt_valid, yt_test = mtb.cross_validation.train_test_split(
                    cv=cv_ot, X=self.Xt, y=self.yt)
                self._share_args(
                    Xt_valid=Xt_valid,
                    Xt_test=Xt_test,
                    yt_valid=yt_valid,
                    yt_test=yt_test)
            else:
    
                Xt_valid, Xt_test, yt_valid, yt_test, groupt_valid, groupt_test = mtb.cross_validation.train_test_split(
                    cv=cv_ai, X=self.Xt, y=self.yt, groups=self.group_t)
                self._share_args(
                    Xt_valid=Xt_valid,
                    Xt_test=Xt_test,
                    yt_valid=yt_valid,
                    yt_test=yt_test,
                    groupt_valid=groupt_valid,
                    groupt_test=groupt_test)

        # model with input parameters
        self._model = GridSearchCV(classifier, parameters, cv=cv_ai)
        
        

        if self.params_ot is None and self.transport_function == ot.da.SinkhornTransport or self.transport_function == ot.da.EMDTransport:
            self.transport_model = self.transport_function()
            self.transport_model.fit(self.Xs, Xt=self.Xt)

        elif self._is_grid_search():
            self._find_best_parameters_crossed(
                self.Xs, ys=self.ys, Xt=self.Xt, yt=self.yt, group_val=self.group_s)
        else :
            self.transport_model = self.transport_function()
            self.transport_model.fit(self.Xs, ys=self.ys, Xt=self.Xt, yt=self.yt)
        
        return self.transport_model

    def predict_transfer(self, data):
        """
        Predict model using domain adaptation.

        Parameters
        ----------
        data : arr.
            Vector to transfer

        Returns
        -------
        data or data_non_scale : arr
            tranfered vector scaled or not scaled

        """

        if self.scaler is not False:
            data = self.Xs_scaler.transform(data)

        data = self.transport_model.transform(Xs=data)
        if self.scaler is not False:
            data_non_scale = self.Xs_scaler.inverse_transform(data)
            return data_non_scale,data #non scale, scale
        else :
            return data #non scale

    def valid_fit_crossed(self, Xs_transform):
        """
        OA comparison before and after OT with Xt_test

        Parameters
        ----------
        Xs_transform : array_like, shape (n_source_samples, n_features)
            Source domain array transformed after OT.

        """

        # avant transport
        self._model.fit(self.Xs, self.ys, self.group_s)
        y_pred_non_transport = self._model.predict(self.Xt_test)
        oa_non_transport = accuracy_score(
            self.yt_test, y_pred_non_transport)
        print("Cross-validation (valid)")
        print("OA before transport :", round(oa_non_transport,3))
        # apres transport
        self._model.fit(Xs_transform, self.ys, self.group_s)
        y_pred_transport = self._model.predict(self.Xt_test)
        oa_transport = accuracy_score(self.yt_test, y_pred_transport)
        print("OA after transport", round(oa_transport,3))
        print("There is a difference of",round(oa_transport-oa_non_transport,4),
              "after transport")


    def assess_transport(self, Xs_transform, record = False, path = None):
        """
        OA comparison before and after OT

        Parameters
        ----------
        Xs_transform : array_like, shape (n_source_samples, n_features)
            Source domain array transformed after OT.

        Returns
        -------
        y_pred_non_transport : array_like
            yt_prediction before transport
        y_pred_transport : array_like
            yt_prediction after transport.

        """

        # avant transport
        self._model.fit(self.Xs, self.ys, self.group_s)
        y_pred_non_transport = self._model.predict(self.Xt)
        oa_non_transport = accuracy_score(self.yt, y_pred_non_transport)
        print("All image")
        print("OA before transport", round(oa_non_transport,3))
        # apres transport
        self._model.fit(Xs_transform, self.ys, self.group_s)
        if record == True : 
            pickle.dump(self._model, open(path, 'wb'))
            print('Learning model has been record')
        y_pred_transport = self._model.predict(self.Xt)
        oa_transport = accuracy_score(self.yt, y_pred_transport)
        print(
            "OA after transport",
            round(oa_transport,3),
            "on all image")
        print(
            "There is a difference of",
            round(
                oa_transport -
                oa_non_transport,
                4),
            "after transport (on all image)")
        
        return y_pred_non_transport, y_pred_transport




    def assess_transport_circular(self,
                                  Xs_transform,
                                  group_s=None,
                                  group_t=None,
                                  cv_ai=StratifiedKFold(
                                      n_splits=2, shuffle=True, random_state=21),
                                  classifier=RandomForestClassifier(random_state=42),
                                  parameters=dict(n_estimators=[100]),
                                  yt = None,
                                  ys=None):
        """
        OA comparison before and after OT

        Parameters
        ----------
        Xs_transform : array_like, shape (n_source_samples, n_features)
            Source domain array transformed after OT.

        Returns
        -------
        y_pred_non_transport : array_like
            yt_prediction before transport
        y_pred_transport : array_like
            yt_prediction after transport.

        """
        self.group_s = group_s
        self.group_t = group_t
        self.yt=yt
        self.ys=ys
        # avant transport
        self._model = GridSearchCV(classifier, parameters, cv=cv_ai)
        self._model.fit(self.Xs, self.ys, self.group_s)
        y_pred_non_transport = self._model.predict(self.Xt)
        oa_non_transport = accuracy_score(self.yt, y_pred_non_transport)
        print("OA before transport", round(oa_non_transport,3))
        # apres transport
        self._model.fit(Xs_transform, self.ys, self.group_s)
        y_pred_transport = self._model.predict(self.Xt)
        oa_transport = accuracy_score(self.yt, y_pred_transport)
        print(
            "OA after transport",
            round(oa_transport,3),
            "sur toute l'image")
        print(
            "There is a difference of",
            round(
                oa_transport -
                oa_non_transport,
                4),
            "after transport (on all image)")
        return y_pred_non_transport, y_pred_transport



    def _share_args(self, **params):
        """
        Allow to stock each parameters enter by the user
        """
        for key, value in params.items():
            if key == 'scaler':
                if value is False:
                    self._need_scale = False
                else:
                    self._need_scale = True

            self.__dict__[key] = value

    def _prefit(self, Xs, Xt):
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
            # permet de stocker le scaler fitté
            self.Xs_scaler = self._to_scale(Xs, self.scaler)
            self.Xs = self.Xs_scaler.transform(Xs)
            self.Xt_scaler = self._to_scale(Xt, self.scaler)
            self.Xt = self.Xt_scaler.transform(Xt)
            print("Xs and Xt are scaled")
        else:
            print("Xs and Xt are not scaled")

    def _to_scale(self, data, method):
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
        # pour vérifier qu'il y a quelque chose dedans après le .fit :
        # scaler.scaler_
        scaler.fit(data)
        return scaler

    def _is_grid_search(self):
        # search for gridSearch
        
        if self.params_ot is not None:
            param_grid = []
            for key in self.params_ot.keys():
                if isinstance(self.params_ot.get(key), (list, np.ndarray)):
                    param_grid.append(key)
            
            self.param_grid = param_grid
            self.params_ot = self.params_ot.copy()

        else :
            self.param_grid = False
        
        if self.param_grid:
            return True
        else:
            return False

    def _generate_params_from_grid_search(self):
        self.param_grids = []
        hyper_param = {key: self.params_ot[key] for key in self.param_grid}
        items = sorted(hyper_param.items())
        keys, values = zip(*items)
        for v in product(*values):
            params_to_add = dict(zip(keys, v))
            self.params_ot.update(params_to_add)
            self.param_grids.append(self.params_ot)

            yield self.params_ot

    def _find_best_parameters_crossed(self, Xs, ys, Xt, yt, group_val):
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
        # boucle qui test chaque hyperparametres
        for gridOT in self._generate_params_from_grid_search():
            print(gridOT)
            # modele de transport pour chaque combinaison de parametres
            transport_model_tmp = self.transport_function(**gridOT)
            # transport
            if self.transport_function == ot.da.SinkhornTransport or self.transport_function == ot.da.EMDTransport :
                transport_model_tmp.fit(Xs=Xs, Xt=Xt)
            elif self.yt_use == False :
                transport_model_tmp.fit(Xs=Xs, Xt=Xt, ys=ys)
            else :
                transport_model_tmp.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
            Xs_transform = transport_model_tmp.transform(
                Xs=Xs)  # transformation des Xs
            # apprentissage du nouveau modele sur Xs_transform
            self._model.fit(Xs_transform, ys, group_val)
            if self.verbose == True :
                print('Crossed validation OA : ' +
                      str(self._model.best_score_))
                print('Best parameter : ' +
                      str(self._model.best_params_))
            # prediction sur les Xt_valid
            yt_pred_valid = self._model.predict(self.Xt_valid)
            oa_transport = accuracy_score(self.yt_valid, yt_pred_valid)
            print("OA after transport", oa_transport)
            print("-------------------------------------------------")
            # meilleurs parametres
            if self.best_score is None or oa_transport > self.best_score:
                self.best_score = oa_transport
                self.best_params = gridOT.copy()  # stocke les meilleurs parametres
                # stocke le modele fitté av les meilleurs parametres = gagne du
                # temps car pas besoin de le refaire à la fin du fit_crossed
                self.transport_model = transport_model_tmp
        if self.verbose:
            print('Best grid is ' +
                  str(self.best_params))
            print('Best score is ' +
                  str(self.best_score))
        print('Best grid is ' + str(self.best_params))

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
            transport_model_tmp.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)

            transp_Xs = transport_model_tmp.inverse_transform(
                Xs=Xt, ys=yt, Xt=Xs, yt=ys)
            # regarde les différences d'aller-retour
            current_score = mean_squared_error(Xs, transp_Xs)

            if self.verbose:
                print(
                    '{} is : {}'.format(self.metrics.__name__, current_score))

            need_update_best_score = False

            if self.best_score is None:
                need_update_best_score = True
            else:
                if self.greater_is_better:
                    # if greater is better, current score need to be higher
                    need_update_best_score = self.best_score < current_score
                else:
                    # if greater is not better, current score need to be lower
                    need_update_best_score = self.best_score > current_score

            if need_update_best_score:  # if need to update best score
                self.best_score = current_score
                # met à jour les meilleurs paramètres si on a obtenu le
                self.best_params = gridOT.copy()
                # meilleur score
                self.transport_model = transport_model_tmp
        if self.verbose:
            print('Best grid is ' +
                  str(self.best_params))
            print('Best score is ' +
                  str(self.best_score))
            
            
    def save_model(self, path):
        """
        Save model 'myModel.npz' to be loaded later via `SuperLearner.load_model(path)`

        Parameters
        ----------
        path : str.
            If path ends with npz, perfects, else will add '.npz' after your fileName.

        Returns
        -------
        path : str.
            Path and filename with mtb extension.
        """
        if not path.endswith('npz'):
            path += '.npz'

        np.savez_compressed(path, SL=self.__dict__)

        return path


    def load_model(self, path):
        """
        Load model previously saved with `SuperLearner.save_model(path)`.

        Parameters
        ----------
        path : str.
            If path ends with npy, perfects, else will add '.npy' after your fileName.
        """
        if not path.endswith('npz'):
            path += '.npz'
        model = np.load(path, allow_pickle=True)
        self.__dict__.update(model['SL'].tolist())




class RasterOptimalTransport(OptimalTransportGridSearch):

    def __init__(self,
                 transport_function=ot.da.MappingTransport,
                 params=None,
                 verbose=True):
        """
        Initialize Python Optimal Transport for raster processing.
        """
        super().__init__(transport_function, params, verbose)



    def preprocessing(self,
                      image_source=None,
                      image_target=None,
                      vector_source=None,
                      vector_target=None,
                      label_source=None,
                      label_target=None,
                      group_source=None,
                      group_target=None,
                      scaler=StandardScaler):
        """
        Scale all image (if it is asked) and stock the input parameters in the object .

        Parameters
        -----------
        image_source : str.
            source image (gdal supported raster) -> path + file name
        image_target : str.
            target image (gdal supported raster) -> path + file name
        vector_source : str.
            labels (gdal supported vector) -> path + file name
        vector_target : str.
            labels (gdal supported vectorR) -> path + file name
        label_source : str.
            name of the label colum in vector file (source)
        label_source : str.
            name of the label colum in vector file (target)
        group_source : str.
            name of the group colum of each polygon in vector file (source)
        group_target : str.
            name of the group colum of each polygon in vector file (target)
        scaler: scale function (default=StandardScaler)
            The function used to scale source and target image

        """

        self._share_args(image_source=image_source,
                         image_target=image_target,
                         vector_source=vector_source,
                         vector_target=vector_target,
                         label_source=label_source,
                         label_target=label_target,
                         group_source=group_source,
                         group_target=group_target,
                         scaler=scaler)


        if self.group_source is None :


            Xs, ys = mtb.processing.extract_ROI(self.image_source,
                                                self.vector_source,
                                                self.label_source)  # Xsource ysource

            Xt, yt = mtb.processing.extract_ROI(self.image_target,
                                                self.vector_target,
                                                self.label_target)  # Xsource ysource

            group_s, group_t = None,None
        else :


            Xs, ys, group_s = mtb.processing.extract_ROI(self.image_source,
                                                                self.vector_source,
                                                                self.label_source,
                                                                self.group_source)  # Xsource ysource

            Xt, yt, group_t= mtb.processing.extract_ROI(self.image_target,
                                                                self.vector_target,
                                                                self.label_target,
                                                                self.group_target)  # Xsource ysource
            
        self._share_args(Xs = self._convert_to_float64(Xs), Xt = self._convert_to_float64(Xt), ys = ys, yt = yt, group_s = group_s, group_t = group_t)

        source_array = mtb.processing.RasterMath(image_source,return_3d=False,
                                          verbose=False).get_image_as_array()

        target_array = mtb.processing.RasterMath(image_target,return_3d=False,
                                          verbose=False).get_image_as_array()


        if self._need_scale :

            self.Xs_non_scaled = Xs
            self.Xt_non_scaled = Xt
            self._prefit_image(source_array,target_array)
            self.Xs = self.source_scaler.transform(self.Xs_non_scaled)
            self.Xt = self.target_scaler.transform(self.Xt_non_scaled)
            print("Image is scaled")

        else :

            self.source = source_array.astype(np.float64)
            self.target = target_array.astype(np.float64)

            print("Image is not scaled")


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


        data = self.transport_model.transform(data)
        if self.scaler is not False:
            data_non_scale = self.source_scaler.inverse_transform(data)
            return data_non_scale,data
        else :
            return data



    def _prefit_image(self, source, target):
        """
        Scale source and target

        Parameters
        ----------
        source : array_like, shape (n_source_samples, n_features)
            Source domain array.
        target : array_like, shape (n_source_samples,)
            Label source array (1d).
        """


        if self._need_scale:
            # permet de stocker le scaler fitté
            self.source_scaler = self._to_scale(source, self.scaler)
            self.source = self.source_scaler.transform(source)
            self.target_scaler = self._to_scale(target, self.scaler)
            self.target = self.target_scaler.transform(target)
            print("source and target are scaled")
        else:
            print("source and target are not scaled")
