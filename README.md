# POTTOK

### Python Optimal Transport for Terrestrial Observation Knowledge

### Détail des fonctions :

#### OptimalTransportGridSearch

- initialise la classe avec la fonction de transport et ses paramètres
- **preprocessing** : enregistre les Xs, ys, Xt, yt dans la classe et les scale si l'utilisateur le demande
- **fit_circular** : cherche les meilleurs paramètres avec la méthode d'aller-retour et fit le modèle de transport avec les meilleurs paramètres 
- **fit_crossed** : cherche les meilleurs paramètres avec la méthode que nous avons déterminée et fit le modèle de transport avec les meilleurs paramètres
- **improve** : sert à évaluer l'apport du transport sur les données tests (utile uniquement après un fit_crossed qui sépare les Xt en valid/test
- **predict_transfert** : sert à transporter les données (les Xs, qui sont rescalée avec le même fit) --> Est-ce qu'on dirait pas à l'utilisateur de directement utiliser les Xs stockés dans la classe comme ça pas besoin de rescaler ?
- _share_args : 
