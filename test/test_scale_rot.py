    # def preprocessing_rm(self,
    #                   in_image_source=None,
    #                   in_image_target=None,
    #                   in_vector_source=None,
    #                   in_vector_target=None,
    #                   in_label_source=None,
    #                   in_label_target=None,
    #                   in_group_source=None,
    #                   in_group_target=None,
    #                   scaler=StandardScaler):
    #     """
    #     Scale all image (if it is asked) and stock the input parameters in the object .

    #     Parameters
    #     -----------
    #     in_image_source : str.
    #         source image (gdal supported raster) -> path + file name
    #     in_image_target : str.
    #         target image (gdal supported raster) -> path + file name
    #     in_vector_source : str.
    #         labels (gdal supported vector) -> path + file name
    #     in_vector_target : str.
    #         labels (gdal supported vectorR) -> path + file name
    #     in_label_source : str.
    #         name of the label colum in vector file (source)
    #     in_label_source : str.
    #         name of the label colum in vector file (target)
    #     in_group_source : str.
    #         name of the group colum of each polygon in vector file (source)
    #     in_group_target : str.
    #         name of the group colum of each polygon in vector file (target)
    #     scaler: scale function (default=StandardScaler)
    #         The function used to scale source and target image
            
    #     """

    #     self._share_args(in_image_source=in_image_source,
    #                      in_image_target=in_image_target,
    #                      in_vector_source=in_vector_source,
    #                      in_vector_target=in_vector_target,
    #                      in_label_source=in_label_source,
    #                      in_label_target=in_label_target,
    #                      in_group_source=in_group_source,
    #                      in_group_target=in_group_target,
    #                      scaler=scaler)
        
    #     Xs, ys, group_s, pos_s = mtb.processing.extract_ROI(self.in_image_source,
    #                                                         self.in_vector_source,
    #                                                         self.in_label_source,
    #                                                         self.in_group_source,
    #                                                         get_pixel_position=True)  # Xsource ysource

    #     Xt, yt, group_t, pos_t = mtb.processing.extract_ROI(self.in_image_target,
    #                                                         self.in_vector_target,
    #                                                         self.in_label_target,
    #                                                         self.in_group_target,
    #                                                         get_pixel_position=True)  # Xsource ysource

    #     self.Xs_non_scale = Xs
    #     self.Xt_non_scale = Xt
        
    #     source_array = mtb.processing.RasterMath(in_image_source,return_3d=False,
    #                                   verbose=False).get_image_as_array()
        
    #     target_array = mtb.processing.RasterMath(in_image_target,return_3d=False,
    #                                   verbose=False).get_image_as_array()
        
        
    #     source_array_test = mtb.processing.RasterMath(in_image_source,return_3d=True,
    #                                  verbose=False).get_image_as_array()
        
    #     target_array_test = mtb.processing.RasterMath(in_image_target,return_3d=True,
    #                                  verbose=False).get_image_as_array()
        
    #     #self._prefit_image(source_array,target_array)
        
    #     self._prefit_image(source_array_test.reshape(*source_array.shape),target_array_test.reshape(*target_array.shape))
        
    #     self.source_3d = self.source.reshape(*source_array_test.shape)
    #     self.target_3d = self.target.reshape(*target_array_test.shape)


    #     self.Xs = self.source_scaler.transform(self.Xs_non_scale)
    #     self.Xt = self.target_scaler.transform(self.Xt_non_scale)
        
    #     Xs_test = self.source_3d[pos_s[:,1].astype(int),pos_s[:,0].astype(int)]
    #     Xt_test = self.target_3d[pos_t[:,1].astype(int),pos_t[:,0].astype(int)]
    
    # (Xs_test == self.Xs).all()