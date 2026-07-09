# Question 6: Construction Zone Detection Module

Ce module fait partie du système ADAS. Il est conçu pour détecter les cônes de chantier orange et les déviations temporaires de trajectoire afin d'informer l'unité centrale de contrôle de la nécessité de désactiver l'aide au maintien de voie standard et de se rabattre à gauche.


## Structure du Module

```text
construction_zone/
├── __init__.py          # Point d'entrée du package Python
├── detector.py          # Logique de traitement d'image et prise de décision
├── demo.py              # Script de test et simulation autonome
├── requirements.txt     # Dépendances du module
└── README.md            
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

