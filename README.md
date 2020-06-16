# POTTOK - v1

### Python Optimal Transport for Terrestrial Observation Knowledge

### Détail des fonctions dans les deux classes :

#### OptimalTransportGridSearch

- initialise la classe avec la fonction de transport et ses paramètres
- **preprocessing** : enregistre les Xs, ys, Xt, yt dans la classe et les scale si l'utilisateur le demande
- **fit_circular** : cherche les meilleurs paramètres avec la méthode d'aller-retour et fit le modèle de transport avec les meilleurs paramètres 
- **fit_crossed** : cherche les meilleurs paramètres avec la méthode que nous avons déterminée et fit le modèle de transport avec les meilleurs paramètres
- **improve** : sert à évaluer l'apport du transport sur les données tests (possible uniquement après un fit_crossed qui sépare les Xt en valid/test)
- **predict_transfert** : sert à transporter les données (les Xs, qui sont rescalée avec le même fit) --> Est-ce que l'utilisateur ne pourrait pas directement utiliser les Xs scalés stockés dans la classe ? 
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

### Ce que j'ai constaté :

En travaillant seulement sur les Xs et les XT, l'utilisation de la classe OptimalTransportGridSearch ne montre pas d'amélioration alors que RasterOptimalTransport montre une petite amélioration. Cela ne semble pas logique car en principe la seule différence est que dans un cas les données ont été  scalées seulement avec Xs et Xt et dans l'autre en fonction de toute l'image.

Il y a aussi un problème, dans rot, comme le scale est fait bande par bande il est impossible d'appliquer l'inverse transform pour avoir les données transportées non scalées. Il faudrait stocker le fit du scaler bande par bande mais cela me semble fastidieux (et je ne suis pas sûre que cela fonctionne sur de grosses images).

Bonne lecture !






