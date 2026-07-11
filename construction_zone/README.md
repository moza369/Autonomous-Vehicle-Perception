# Question 6: Construction Zone Detection Module

This module uses a classical computer vision pipeline based on HSV color segmentation, morphological filtering, contour detection, and geometric analysis to detect orange construction cones. When multiple cones are detected, the system identifies a construction zone and displays the required ADAS action.

## Structure du Module

```text
construction_zone/
├── main.py
├── detector.py
├── requirements.txt
├── README.md
├── test_images/
│   └── construction_zone_real.jpg
└── results/
    └── output_construction_zone_real.png         
```


## Installation

Pour installer les dépendances nécessaires à ce module, exécutez la commande suivante à la racine du projet : pip install -r requirements.txt


## Format des Données de Sortie (API)

La méthode `process_image` retourne trois éléments :

1. **`annotated_frame`** : `np.ndarray` (image BGR) prête à être affichée à l'écran.
2. **`merge_required`** : `bool` (vrai si une ligne de cônes orange coupe ou dévie la trajectoire du véhicule vers la gauche).
3. **`metadata`** : `dict` contenant :
   * `'detected_cones_count'` (`int`) : Le nombre de cônes valides actuellement détectés.
   * `'cones_coordinates'` (`list[tuple[int, int]]`) : Coordonnées pixel $(x, y)$ au sol des cônes détectés.
   * `'fitted_line'` (`tuple[tuple[int,int], tuple[int,int]]` ou `None`) : Points $(x1, y1)$ et $(x2, y2)$ délimitant la ligne virtuelle ajustée.
   * `'slope'` (`float`) : Pente de la droite ajustée (sert à mesurer la sévérité de l'empiètement).

## Run the real construction zone test

```bash
python main.py



