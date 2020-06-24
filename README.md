# POTTOK 

### Python Optimal Transport for Terrestrial Observation Knowledge

### Détail des fonctions dans les deux classes :

#### OptimalTransportGridSearch

- initialise la classe avec la fonction de transport et ses paramètres
- **preprocessing** : enregistre les Xs, ys, Xt, yt dans la classe et les scale si l'utilisateur le demande
- **fit_circular** : cherche les meilleurs paramètres avec la méthode d'aller-retour et fit le modèle de transport avec les meilleurs paramètres 
- **fit_crossed** : cherche les meilleurs paramètres avec la méthode que nous avons déterminée et fit le modèle de transport avec les meilleurs paramètres
- **predict_transfert** : sert à transporter les données (les Xs, qui sont rescalée avec le même fit) --> Est-ce que l'utilisateur ne pourrait pas directement utiliser les Xs scalés stockés dans la classe ? 
- **assess_transport** : sert à évaluer l'apport du transport sur les données tests (possible uniquement après un fit_crossed qui sépare les Xt en valid/test)
- **_share_args** : stocke les paramètres rentrés dans l'objet 
- **_prefit** : scale Xs et Xt
- **_to_scale** : fit les données rentrées avec une fonction scaler choisie par l'utilisateur
- **_is_grid_search** : détermine si il y a besoin d'une grille de recherche
- **_generate_params_from_grid_search** : sort chaque combinaison de paramètre possible
- **_find_best_parameters_crossed** : applique la méthode utilisée dans **fit_crossed**
- **_find_best_parameters_circular** : applique la méthode utilisée dans **fit_circular**

#### RasterOptimalTransport

- l'initialisation est la même que pour OptimalTransportGridSearch
- **preprocessing** : scale ou non les images sources et cibles rentrées par l'utilisateur, extrait les Xs, ys, Xt, yt de ces images scalées ou non. 
- **predict_transfert** : transporte les données. les données rentrées ne sont pas de nouveaux scalées contrairement à la fonction de otgs ce qui signifie que l'utilisateur doit bien faire attention à prendre les données enregistrées dans la classe (.image_scale_source_reshape)
- **im2mat**
- **mat2im**
- **_to_scale** : scale les données entrées. 








