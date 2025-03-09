# Scoring, Classification et Visualisation à partir de fichier lidar

Le **LiDAR (Light Detection And Ranging)** est une technologie de télédétection active qui permet de cartographier en 3D le sol et le sursol. En France, l’IGN diffuse en open data des nuages de points LiDAR HD couvrant tout le territoire via la plateforme [diffusion-lidarhd.ign.fr](https://diffusion-lidarhd.ign.fr/). Ces nuages de points sont généralement fournis au format LAS/LAZ et contiennent des millions de points 3D classifiés (sol, végétation, bâtiments, etc.). 

Dans ce tutoriel, nous allons expliquer comment traiter un fichier LiDAR issu de l’IGN avec des outils open-source en Python. L’objectif est de **télécharger des données brutes**, de les **préparer**, puis d’**extraire des indicateurs** permettant de caractériser le paysage (de façon simple, sans prétention scientifique). Enfin, nous verrons comment **visualiser les résultats** pour interpréter ces indicateurs. 

*Au programme :* téléchargement d’une dalle LiDAR, lecture du nuage de points, calcul d’indices (densité de bâtiments, végétation, rugosité du sol, etc.) et attribution d’un **score de paysage** par petite zone (par exemple « urbain », « forêt », « champs »…), puis affichage des zones classées sur une carte. Ce tutoriel s’adresse aux débutants curieux de manipuler des données LiDAR HD en open-source.

# Récupération et préparation des données

## Téléchargement des données LiDAR IGN

La plateforme de diffusion LiDAR HD de l’IGN permet de télécharger des dalles de nuages de points. Chaque dalle couvre typiquement **1 km²** et est disponible en format compressé `.laz`. On peut récupérer une dalle soit via l’interface web (en cherchant la tuile correspondante sur la carte), soit directement par une URL. Dans notre exemple, nous utiliserons une URL connue pour télécharger une dalle de démonstration.

Commençons par importer les bibliothèques nécessaires, notamment `laspy` pour lire les fichiers LAS/LAZ, et `requests` pour le téléchargement : 

```python
!pip install laspy[lazrs, laszip]  # installation de laspy et de ses extensions de décompression
import laspy
import requests
import numpy as np
```

> **Note :** Le package **laspy** permet de lire les fichiers `.laz` compressés grâce à l’extension `lazrs`. Veillez à installer ces dépendances si nécessaire.

Une fois les bibliothèques prêtes, téléchargeons le fichier. On définit l’URL de la dalle et on enregistre le contenu binaire dans un fichier local (par exemple `data.laz`) :

```python
url = "https://storage.sbg.cloud.ovh.net/.../LHD_FXX_0938_6673_PTS_C_LAMB93_IGN69.copc.laz"
nom_fichier = "data.laz"

# Télécharger le fichier en streaming par chunks pour éviter la surcharge mémoire
response = requests.get(url, stream=True)
with open(nom_fichier, "wb") as f:
    for chunk in response.iter_content(chunk_size=157286400):  # ~150 MB
        if chunk:
            f.write(chunk)
```

Une fois le fichier téléchargé, on peut le **lire avec laspy** :

```python
las = laspy.read(nom_fichier)
```

La variable `las` contient maintenant le nuage de points et ses attributs. 

## Exploration des données brutes

Avant d’analyser les points, examinons quelques informations de base du fichier : 

- **Entête (header)** : contient le SRS (système de coordonnées), l’altitude de référence, etc.
- **Nombre de points total**
- **Format des points** : dimensions enregistrées pour chaque point (coordonnées X, Y, Z, intensité, classification, etc.)

On peut afficher ces informations ainsi :

```python
print("=== En-tête du fichier ===")
print(las.header)

print("\n=== Nombre total de points ===")
print(len(las.points))

print("\n=== Format des points ===")
print(las.point_format)            # décrit le format binaire des points

print("\n=== Dimensions disponibles par point ===")
print(las.point_format.dimension_names)
```

Par exemple, l’entête nous indique le système de projection utilisé. Ici, les données sont en coordonnées projetées **EPSG:2154 (Lambert-93)** avec des altitudes en **IGN69** (niveau NGF). Les noms de dimensions confirment que chaque point a des attributs comme `X, Y, Z, intensity, classification, return_number` etc. Le **nombre de points** d’une dalle LiDAR 1 km² peut être très élevé (plusieurs dizaines de millions de points). Dans notre cas de démonstration, on a ~24 millions de points. 

Chaque point LiDAR est également **classifié** par l’IGN (sol, bâtiment, végétation, eau, etc.). Par exemple, la classe 2 correspond généralement au sol, 6 aux bâtiments, 3-5 à la végétation (basse, moyenne, haute), etc. Ces classifications nous seront utiles pour l’analyse.

## Préparation de l’échantillon de travail

Manipuler 24 millions de points peut être lourd en mémoire et en calcul. Pour l’illustration, nous allons **extraire un échantillon aléatoire** de points afin de travailler plus rapidement. 

Fixons par exemple un échantillon de `100000` points (ce qui représente environ 0,4% des points dans notre cas) :

```python
sample_size = 100000  # taille de l'échantillon
total_points = len(las.x)
indices = np.random.choice(total_points, size=sample_size, replace=False)
print(f"Taille de l'échantillon : {len(indices)}, soit {len(indices)/total_points*100:.2f}% des points")
```

On obtient un échantillon de 100k points, ce qui est suffisant pour calculer des indicateurs globaux sur la dalle. On extrait alors les données de ces indices :

```python
# Extraction des champs utiles pour les points échantillonnés
x = np.array(las.x[indices])
y = np.array(las.y[indices])
z = np.array(las.z[indices])
intensity = np.array(las.intensity[indices])
classification = np.array(las.classification[indices])
return_number = np.array(las.return_number[indices])
```

Nous avons maintenant des tableaux `x, y, z` des coordonnées des points échantillonnés, et les attributs d’intensité, classification, etc., correspondants. On peut par exemple afficher quelques lignes de ces données sous forme de table pour vérifier :

```python
import pandas as pd
df = pd.DataFrame({
    'x': x, 'y': y, 'z': z,
    'classification': classification,
    'intensity': intensity,
    'return_number': return_number
})
print(df.head())
```

Cela affichera quelque chose comme :

```
           x           y       z  classification  intensity  return_number
0  938534.06  6672207.19  332.54               5       1224               1
1  938077.56  6672403.83  351.62               2        463               2
2  938187.66  6672056.38  330.26               2       1132               1
3  938762.01  6672675.91  396.44               3        192               2
4  938984.57  6672217.78  426.87               2       1177               1
```

Chaque ligne correspond à un point. On voit par exemple le point 0 de l’échantillon a des coordonnées (x=938534.06, y=6672207.19, z=332.54) et est classé `5` (ce qui pourrait correspondre à de la **haute végétation**), avec une intensité de 1224 et c’est le **premier retour** d’un tir laser (return_number=1). Le point 1 est classé `2` (probablement **sol**). Ces données brutes nous serviront à calculer des indicateurs.

Avant de passer à l’analyse, notons la **zone géographique** couverte par ces points. Les coordonnées étant en Lambert-93 (mètres), on peut trouver l’emprise minimale et maximale :

```python
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
print(x_min, x_max, y_min, y_max)
```

Dans notre exemple, x ~ [938000, 939000] et y ~ [6672000, 6673000], ce qui correspond bien à une dalle de 1 km². Si besoin, on peut convertir ces coordonnées en latitude/longitude pour situer la zone. Par exemple, avec pyproj/folium, on peut afficher la zone sur une carte interactive :

```python
from pyproj import Transformer
import folium

# Transformer Lambert-93 -> WGS84 (lat-long)
transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
min_lon, min_lat = transformer.transform(x_min, y_min)
max_lon, max_lat = transformer.transform(x_max, y_max)

# Carte centrée sur la zone
centre_lat = (min_lat + max_lat) / 2
centre_lon = (min_lon + max_lon) / 2
m = folium.Map(location=[centre_lat, centre_lon], zoom_start=15)
# Rectangle de la tuile
folium.Rectangle(bounds=[(min_lat, min_lon), (max_lat, max_lon)],
                 color="blue", fill=True, fill_opacity=0.2).add_to(m)
m
```

Cette étape est optionnelle, mais elle permet de vérifier la localisation de la dalle (par exemple, on s’assure que ce n’est pas en plein océan ou une zone interdite). 

Nous avons maintenant nos données prêtes pour l’analyse : un sous-ensemble de points `x,y,z` avec des classes.

# Analyse et scoring

L’objectif de l’analyse est de **caractériser le paysage** de la dalle en la découpant en petites zones et en calculant des **indicateurs** simples sur chaque zone. Ensuite, nous attribuerons un *score* à chaque zone pour différents types de paysages (urbain, rural, forestier, etc.) en nous basant sur ces indicateurs.

## Découpage en grille

Nous allons diviser la dalle de 1 km² en une **grille de petites cellules** pour analyser localement le paysage. Choisissons par exemple des carreaux de **25 m de côté**. Avec 1000 m de côté pour la dalle, on obtient une grille de 40 x 40 cellules (1600 cellules au total). 

Définissons la taille de cellule et calculons le nombre de cellules sur X et Y :

```python
cell_size = 25  # taille d'une cellule (25 m)
nx = int(np.ceil((x_max - x_min) / cell_size))
ny = int(np.ceil((y_max - y_min) / cell_size))
print(nx, ny)  # devrait donner ~40, 40
```

Nous pouvons créer des matrices pour stocker les résultats par cellule :

```python
grid_classification = np.empty((ny, nx), dtype=object)  # classification finale (type de paysage) par cellule
grid_score = np.empty((ny, nx), dtype=object)           # scores détaillés par cellule
```

## Calcul des indicateurs par cellule

Parcourons chaque cellule de la grille et **sélectionnons les points LiDAR qui tombent à l’intérieur**. À partir de ces points, nous calculerons différents indicateurs. Pour chaque cellule de coordonnées d’indice `(i,j)` :

1. Déterminer les bornes de la cellule en X et Y.
2. Extraire les points dont `x` et `y` sont dans ces bornes (un simple masque booléen).
3. Si la cellule contient suffisamment de points (par ex. au moins 10 points pour la fiabilité statistique), calculer les indicateurs.

Voici à quoi ressemble la boucle principale de calcul :

```python
for i in range(ny):
    for j in range(nx):
        # Bornes de la cellule (en coordonnées projetées)
        cell_x_min = x_min + j * cell_size
        cell_x_max = cell_x_min + cell_size
        cell_y_min = y_min + i * cell_size
        cell_y_max = cell_y_min + cell_size

        # Masque des points appartenant à cette cellule
        mask = (x >= cell_x_min) & (x < cell_x_max) & (y >= cell_y_min) & (y < cell_y_max)
        n_points = np.sum(mask)
        if n_points < 10:
            # Trop peu de points dans la cellule -> on classe comme "Insuffisant"
            grid_classification[i, j] = "Insuffisant"
            grid_score[i, j] = "Insuffisant"
            continue

        # À partir d'ici, on a suffisamment de points pour calculer les indicateurs...
        # (calculs détaillés ci-dessous)
```

À l’intérieur de cette boucle, pour chaque cellule avec suffisamment de points, nous calculons plusieurs **indicateurs caractéristiques** du terrain :

- **Densité de points** (`density`) : nombre de points par m² dans la cellule (utile pour repérer zones vides ou très denses).
- **Proportion de bâtiments** (`prop_batiments`) : fraction de points classés comme bâtiment (code classification 6) parmi les points de la cellule.
- **Proportion de végétation** (`prop_vegetation`) : fraction de points classés végétation (codes 3,4,5) dans la cellule.
- **Proportion de sol** (`prop_ground`) : fraction de points classés sol (code 2).
- **Étendue d’altitude du sol** (`range_sol`) : différence entre l’altitude max et min des points sol dans la cellule (si pas de sol, on prendra l’ensemble des points).
- **Rugosité du sol** (`rugosite_sol`) : écart-type des altitudes des points sol (mesure la variabilité du terrain).
- **Hauteur moyenne** (`mean_height`) : moyenne des altitudes des points de la cellule (globalement, hauteur moyenne du couvert).
- **Écart interquartile de hauteur** (`iqr_height`) : différence entre le 3e et le 1er quartile des hauteurs des points (mesure de la dispersion des hauteurs, indicateur de végétation de hauteurs variées par ex).
- **Indice de fragmentation des classes** (`fragmentation_index`) : nombre de classes de points distinctes présentes dans la cellule, divisé par le nombre de points (plus il y a de classes différentes de points, plus le milieu est hétérogène).
- **Ratio bâtiment/sol** (`building_to_ground_ratio`) : nombre de points bâtiment sur nombre de points sol (indique l’importance relative des bâtiments par rapport au sol nu).
- **Pente moyenne** (`pente_moyenne`) : pente estimée du terrain dans la cellule. Pour la calculer, on peut ajuster un plan aux points (par régression linéaire de z en fonction de x,y) et en déduire l’inclinaison.
- **Indice de planéité** (`indice_planeite`) : écart-type des résidus par rapport au plan ajusté (plus c’est proche de 0, plus le terrain s’ajuste bien à un plan -> surface plane; plus c’est grand, plus le relief est accidenté même après retrait de la pente).
- **Indice d’anisotropie** (`anisotropy_index`) : indicateur de la forme des données au sol, obtenu via une ACP (Analyse en Composantes Principales) sur les coordonnées (x,y) des points de la cellule. On peut par exemple prendre le ratio des deux valeurs propres principales de la PCA : un ratio élevé signifie que la répartition des points est étirée dans une direction (ex: alignement linéaire, route, haie), tandis qu’un ratio proche de 1 indique une distribution isotrope (plutôt circulaire ou uniforme dans la cellule).
- **Intensité moyenne** (`mean_intensity`) : intensité lidar moyenne des points (peut donner des infos sur la nature de la surface, réfléchissante ou non).
- **Ratio de retours uniques** (`single_return_ratio`) : proportion de points pour lesquels le laser n’a eu qu’un seul retour (par opposition à plusieurs échos). Par exemple, un sol nu donnera souvent un seul retour, alors qu’une végétation dense peut donner plusieurs retours (feuillage puis sol).

Voici comment on peut implémenter quelques-uns de ces calculs dans la boucle :

```python
        # Calcul des indicateurs dans la cellule
        z_cell = z[mask]
        # Proportions par classe
        prop_batiments = np.sum(classification[mask] == 6) / n_points
        prop_vegetation = np.sum(np.isin(classification[mask], [3,4,5])) / n_points
        prop_ground = np.sum(classification[mask] == 2) / n_points

        # Altitudes 
        alt_range = z_cell.max() - z_cell.min()
        rugosite = np.std(z_cell)
        mean_height = np.mean(z_cell)
        iqr_height = np.percentile(z_cell, 75) - np.percentile(z_cell, 25)

        # Fragmentation des classes
        nb_classes = len(np.unique(classification[mask]))
        fragmentation_index = nb_classes / n_points

        # Ratio Bâtiment/Sol
        if np.sum(classification[mask] == 2) > 0:
            building_to_ground_ratio = np.sum(classification[mask] == 6) / np.sum(classification[mask] == 2)
        else:
            building_to_ground_ratio = 0
```

Pour la **pente moyenne** et l’**indice de planéité**, on ajuste un plan aux points (x,y,z) via une régression plane :

```python
        # Calcul de la pente via régression plane
        coords = np.vstack((x[mask], y[mask])).T  # matrice des coordonnées au sol
        reg = LinearRegression().fit(coords, z_cell)
        # Les coefficients reg.coef_ donnent les pentes suivant x et y
        grad_x, grad_y = reg.coef_
        # Norme du gradient
        grad = np.sqrt(grad_x**2 + grad_y**2)
        # Pente en degrés
        slope = np.degrees(np.arctan(grad))

        # Planéité : écart-type des résidus du plan
        z_pred = reg.predict(coords)
        residus = z_cell - z_pred
        planarity = np.std(residus)
```

Et pour l’**anisotropie** via PCA sur la distribution horizontale des points :

```python
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(coords)
        # Ratio des deux composantes principales
        anisotropy_index = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
```

Enfin, pour l’**intensité moyenne** et le **ratio de retours uniques** :

```python
        mean_intensity = np.mean(intensity[mask])
        single_return_ratio = np.sum(return_number[mask] == 1) / n_points
```

À ce stade, nous avons un ensemble d’indicateurs numériques pour la cellule en question. Bien sûr, d’autres indicateurs pourraient être calculés selon les besoins, mais cette liste couvre différents aspects : occupation du sol (bâtiment/végétation), topographie (pente, altitude), structure 3D (hauteurs, anisotropie) et propriétés du signal (intensité, retours).

## Fonction de scoring des paysages

Maintenant que nous avons des *descripteurs* pour chaque cellule, comment déterminer le type de paysage dominant dans cette cellule ? Pour cela, nous définissons de manière **heuristique** une fonction de scoring qui attribue un score pour différents types de paysages en fonction des indicateurs. 

Dans notre exemple, imaginons que nous voulons distinguer les catégories suivantes :
- **Ville** (zone urbaine dense)
- **Village / Banlieue** (zone bâtie moins dense)
- **Champs** (zone ouverte, agricole, plane)
- **Forêt** (zone boisée)
- **Montagne** (zone naturelle à fort relief)
- **Eau** (zone recouvrant des surfaces d’eau)
- (Également la catégorie **Insuffisant** pour les cellules avec peu de données, que nous avons déjà marquée)

La stratégie de scoring est simple et *non scientifique* : on va utiliser des règles empiriques et des pondérations pour chaque catégorie. Par exemple :
- Une cellule sera **urbaine (Ville)** si elle a **beaucoup de points de bâtiment**, une certaine hauteur moyenne (immeubles) et un ratio bâtiment/sol élevé. On peut donner un score Ville = `500 * prop_batiments + 50 * mean_height + 50 * building_to_ground_ratio` (juste un exemple de pondération).
- Pour un **Village**, on baisse un peu les exigences : un peu de bâtiments suffit (seuil plus bas) et on pondère différemment : `score_village = 400 * prop_batiments + 30 * mean_height + 25 * building_to_ground_ratio`.
- Une cellule de **Champs** sera caractérisée par *peu ou pas de bâtiments*, *peu de végétation*, et un terrain *plat*. On peut par exemple partir d’un score de base inversément proportionnel aux bâtiments et végétation, et ajouter des bonus si la pente est faible et le terrain plan : `score_champs = 50*(1 - prop_batiments) + 50*(1 - prop_vegetation) + bonus_slope + bonus_planarity`, où `bonus_slope` pourrait valoir par exemple `5 * max(0, 5 - pente_moyenne)` (qui donne des points si la pente_moyenne est en dessous de 5°) et `bonus_planarity = 2 * max(0, 10 - indice_planeite)` (qui donne des points si le terrain est très plan, faible rugosité).
- Une cellule de **Forêt** aura beaucoup de végétation et une certaine hétérogénéité en hauteur. On peut mettre `score_foret = 800 * prop_vegetation + 50 * fragmentation_index + 20 * iqr_height`. Ici on considère qu’une forte proportion de végétation donne un score élevé, augmenté si la diversité des classes est élevée (signe d’un milieu riche, sol + végétation de plusieurs strates) et si l’écart de hauteur est grand (forêt avec canopée élevée).
- Une cellule de **Montagne** sera avantagée si le relief est important (grande étendue d’altitude, pente forte, rugosité forte), mais pénalisée si des bâtiments sont présents (car alors c’est de l’urbain en montagne). Par exemple : `score_montagne = 10*range_sol + 10*rugosite_sol + 15*pente_moyenne + 5*indice_planeite + 30*anisotropy_index + 20*iqr_height - 1000*prop_batiments`, le tout activé seulement si `range_sol` > 100 m **ou** pente > 8° (sinon score_montagne = 0, car zone pas suffisamment « montagneuse »).
- Pour l’**Eau**, on pourrait baser le score sur la proportion de points classés eau, la très faible intensité retournée sous l’eau, etc. (Dans notre code exemple, la catégorie Eau était prévue mais nous n’avions pas forcément de plan d’eau sur la dalle test, donc le score n’a pas été implémenté explicitement et reste à 0 par défaut).

**Important :** Ces formules sont *arbitraires et purement illustratives*. En pratique, un véritable modèle de classification de paysages nécessiterait une approche plus rigoureuse (par exemple de l’apprentissage supervisé sur des zones d’entraînement connues, ou au minimum des règles basées sur des connaissances expertes calibrées). Mais pour notre tutoriel, l’idée est de montrer comment on peut combiner ces indicateurs pour aboutir à une décision de classification.

Voici à quoi ressemble la fonction de score définie dans le notebook :

```python
def scorer_paysages(prop_batiments, prop_vegetation, range_sol, rugosite_sol,
                    mean_height, iqr_height, fragmentation_index,
                    building_to_ground_ratio, anisotropy_index,
                    pente_moyenne, indice_planeite, prop_ground, 
                    single_return_ratio, mean_intensity):
    """
    Calcule un score pour plusieurs types de paysage à partir des indicateurs fournis.
    Retourne un dictionnaire avec les scores pour chaque type de paysage.
    """
    scores = {
        "Ville": 0,
        "Village / Banlieue": 0,
        "Champs": 0,
        "Forêt": 0,
        "Montagne": 0,
        "Eau": 0
    }
    # Score Ville
    if prop_batiments >= 0.1:
        scores["Ville"] = 500*prop_batiments + 50*mean_height + 50*building_to_ground_ratio
    # Score Village/Banlieue
    if prop_batiments >= 0.01:
        scores["Village / Banlieue"] = 400*prop_batiments + 30*mean_height + 25*building_to_ground_ratio
    # Score Champs (avec bonus si plat)
    bonus_slope = max(0, 5 - pente_moyenne) * 5
    bonus_planarity = max(0, 10 - indice_planeite) * 2
    scores["Champs"] = 50*(1 - prop_batiments) + 50*(1 - prop_vegetation) + bonus_slope + bonus_planarity
    # Score Forêt
    scores["Forêt"] = 800*prop_vegetation + 50*fragmentation_index + 20*iqr_height
    # Score Montagne
    if range_sol >= 100 or pente_moyenne >= 8:
        scores["Montagne"] = (10*range_sol + 10*rugosite_sol + 15*pente_moyenne + 
                               5*indice_planeite + 30*anisotropy_index + 20*iqr_height - 1000*prop_batiments)
    return scores
```

On appelle cette fonction pour chaque cellule avec les indicateurs calculés. Elle renvoie un dictionnaire de scores. Nous choisissons ensuite la catégorie ayant le score le plus élevé comme classification finale de la cellule :

```python
scores = scorer_paysages(prop_batiments, prop_vegetation, alt_range, rugosite,
                         mean_height, iqr_height, fragmentation_index,
                         building_to_ground_ratio, anisotropy_index,
                         slope, planarity, prop_ground, single_return_ratio, mean_intensity)
cell_class = max(scores, key=scores.get)  # clé du max du dictionnaire de scores
grid_classification[i, j] = cell_class
grid_score[i, j] = scores
```

Ainsi, `grid_classification` est remplie petit à petit avec des étiquettes comme "Champs", "Forêt", "Village / Banlieue", etc., pour chaque cellule de 25 m de côté.

Une fois la double boucle terminée, nous avons classé chaque portion de la dalle LiDAR selon le type de paysage dominant estimé. 

# Visualisation des résultats

Après tout ce traitement, il est temps de **visualiser la grille de classification** obtenue, ainsi que de confronter ces résultats aux données d’origine pour en évaluer la cohérence.

## Carte de classification par cellule

On peut produire une visualisation 2D simple de la grille, en attribuant une couleur par catégorie de paysage. Par exemple : rouge pour "Ville", orange pour "Village/Banlieue", jaune pour "Champs", vert foncé pour "Forêt", noir pour "Montagne", bleu pour "Eau", et gris pour "Insuffisant". 

Utilisons Matplotlib pour afficher une image de la grille classifiée :

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Palette de couleurs pour chaque classe
color_map = {
    "Ville": "red",
    "Village / Banlieue": "orange",
    "Champs": "yellow",
    "Forêt": "darkgreen",
    "Montagne": "black",
    "Eau": "blue",
    "Insuffisant": "gray"
}

# Construire une image RGB de la grille de classification
grid_rgb = np.zeros((ny, nx, 3))
for i in range(ny):
    for j in range(nx):
        # Convertir la couleur nommée en valeur RGB
        col = mcolors.to_rgb(color_map.get(grid_classification[i, j], "white"))
        grid_rgb[i, j, :] = col

plt.figure(figsize=(8, 8))
plt.imshow(grid_rgb, origin='lower', extent=(x_min, x_max, y_min, y_max))
plt.colorbar()  # barre des couleurs (elle ne correspond pas directement aux classes ici, c’est indicatif)
plt.title("Classification du paysage par cellule (25m)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()
```

Sur cette carte raster, chaque pixel de 25 m prend la couleur de la classe dominante dans cette cellule. Si l’on superpose mentalement sur la zone, on peut identifier des zones continues de même couleur qui devraient correspondre à des entités géographiques : par ex. un massif forestier en vert, une plaine agricole en jaune, un village en orange, etc.

Bien sûr, l’image générée par Matplotlib est une simple visualisation *dans le repère des coordonnées projetées*. On peut enrichir la visualisation de plusieurs façons : 

- Ajouter une **légende** qui mappe les couleurs aux labels (ici on sait que rouge=Ville, etc., on pourrait dessiner un petit patch par catégorie).
- Superposer cette grille sur une **carte** (par exemple, utiliser folium ou contextily pour afficher la grille sur fond de carte IGN ou OpenStreetMap). Dans ce cas, il faudrait reprojeter les coins des cellules en lat/lon et créer des polygones colorés semi-transparents ou un calque image georéférencé.
- Utiliser une **visualisation interactive** pour inspecter les résultats : par exemple Plotly pour afficher la grille et voir en infobulle le détail des scores de chaque cellule.

Dans le notebook, un exemple *bonus* a été réalisé avec Plotly : on crée un Heatmap où chaque cellule est codée par une valeur numérique de classe, et le texte au survol affiche les scores en pourcentage pour chaque catégorie de la cellule. Cela permet en survolant la carte de savoir que telle case était par exemple 70% forêt, 30% champs selon les scores calculés. 

*(Ce niveau de détail dépasse le cadre de notre tutoriel, mais sachez que c’est possible : on peut ainsi combiner les avantages de l’interactivité web avec l’analyse Python.)*

## Visualisation des données brutes vs classification

Pour évaluer la qualité de notre classification, il peut être intéressant de **comparer avec les données LiDAR brutes**. Par exemple : 

- Afficher un **nuage de points 3D** de l’échantillon coloré selon la classification du point fournie par l’IGN (sol en brun, végétation en vert, bâtiments en rouge, etc.), à côté de notre grille de paysages agrégés.
- Regarder sur une carte aérienne ou cadastrale de la zone pour voir si les zones identifiées comme “Village” correspondent bien à un village réel, etc.

Dans le notebook, nous avons tracé une figure 3D Matplotlib de l’échantillon de points (x,y,z) colorés par leur classe d’origine pour un aperçu visuel du terrain. On peut également tracer des coupes 2D ou des histogrammes d’altitude par classe pour mieux comprendre le relief.

Pour rester dans une vue 2D simple, on pourrait par exemple tracer la **projection au sol des points de l’échantillon** en leur attribuant une couleur par classe LiDAR, et superposer la grille de classification par-dessus. Si notre grille “Champs/Forêt/Village” correspond bien, on s’attend par exemple à ce que la zone où nous avons mis “Village” corresponde à un cluster de points bâtiments (classe 6) sur la projection, etc.

Voici une approche simple de comparaison : créer un scatter plot des points au sol et y superposer la grille semi-transparente. Cependant, étant donné le grand nombre de points même échantillonnés, ce scatter peut être lourd à afficher. Une alternative consiste à **rasteriser** aussi les points (par exemple en densité par pixel) pour les afficher en background.

## Résultats obtenus

Dans notre cas d’étude, la dalle choisie se situe (par hypothèse) dans une zone semi-rurale avec un petit village et des forêts. Le résultat de la classification par cellule pourrait ressembler à ceci :
- Des cellules **“Village/Banlieue”** (orange) groupées à l’endroit du bourg (quelques bâtiments).
- Autour, des cellules majoritairement **“Champs”** (jaune) dans les zones de plaine agricole sans beaucoup de végétation.
- Sur les hauteurs ou zones boisées, des cellules **“Forêt”** (vert foncé).
- Éventuellement sur les pentes très fortes, quelques cellules classées **“Montagne”** (noir) si les seuils de pente/relief ont été atteints.
- Pas de “Ville” (rouge) car pas de zone urbaine dense sur cette dalle, et pas de “Eau” (bleu) si pas de lac ou rivière assez grand capté.

L’analyse montre ainsi comment, à partir de données ponctuelles brutes, on a pu extraire des informations thématiques. La visualisation finale, bien qu’approximative, segmente le paysage en différentes zones cohérentes.

# Conclusion

Pour récapituler, nous avons parcouru les étapes suivantes dans ce tutoriel :

1. **Téléchargement de données LiDAR HD** sur le site de l’IGN et lecture du nuage de points avec des outils open-source (laspy).
2. **Préparation des données** pour l’analyse : sélection d’un échantillon de points pour faciliter le prototypage, extraction des champs utiles (coordonnées, classes, intensité…).
3. **Calcul d’indicateurs** par petite zone (25m x 25m) : densités, proportions de classes, statistiques d’altitudes, indices de forme et de relief, etc.
4. **Définition d’une fonction de scoring** simple pour différencier des types de paysages (urbain, rural, forêt, montagne…) sur la base de ces indicateurs.
5. **Application du scoring** à chaque cellule de la grille afin de classifier l’ensemble de la dalle.
6. **Visualisation des résultats** sous forme de carte thématique (raster coloré) et comparaison visuelle avec les données sources.

Ce projet démontre qu’il est possible de faire de l’**analyse de paysage automatisée à partir de données LiDAR ouvertes**, en utilisant uniquement des bibliothèques Python gratuites. Bien sûr, les règles de décision employées ici sont simplifiées. Pour aller plus loin, on pourrait :

- Affiner ou **calibrer les scores** à l’aide de données de référence (par exemple, comparer avec une carte d’occupation du sol connue pour ajuster les pondérations ou entraîner un modèle automatique).
- Exploiter **tous les points** au lieu d’un échantillon pour une meilleure précision spatiale (en optimisant le code pour supporter des dizaines de millions de points, via des techniques de spatial join efficace ou en utilisant des données raster dérivées comme MNT/MNS).
- Intégrer la **troisième dimension de visualisation** plus systématiquement : par exemple produire des vues 3D colorisées pour chaque classe de paysage détecté, ou draper la grille de classification sur un modèle numérique de terrain.
- Tester sur d’autres dalles aux caractéristiques différentes (zone urbaine dense, zone littorale, haute montagne…) pour voir comment se comportent les indicateurs et ajuster éventuellement les formules.
- Utiliser des outils SIG dédiés (QGIS, Whitebox, etc.) pour valider nos résultats en les comparant à des approches existantes de classification LiDAR.

En conclusion, le traitement de fichiers LiDAR offre un vaste champ d’exploration. Grâce à l’open data et aux outils open-source, il est à la portée de tous d’expérimenter et d’extraire de l’information géographique de ces nuages de points massifs. Ce tutoriel vous aura, espérons-le, donné un aperçu de la démarche et l’envie de pousser plus loin l’analyse des paysages à partir des données LiDAR HD de l’IGN. Bonne cartographie 3D !
