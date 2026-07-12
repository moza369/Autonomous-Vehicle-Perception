#  TSR — Traffic Sign and Signal Recognition

### Système de perception et de prise de décision par Deep Learning pour la conduite autonome

**Module :** Computer Vision — Master MIATE
**Scénario :** Team Collaboration & Autonomous Vehicle Perception — Scénario 3 (TSR)

---

##  Aperçu du projet

Ce notebook implémente un pipeline complet et entraînable de deep learning qui :

1. **Détecte** les panneaux de limitation de vitesse, les panneaux stop et l'état des feux
   de signalisation (rouge / jaune / vert) à l'aide d'un détecteur d'objets
   **YOLOv8 / YOLO11** (Ultralytics, PyTorch) entraîné from scratch sur des jeux de données
   réels — **aucune vision par ordinateur classique** (seuillage HSV, cercles de Hough,
   contours, template matching) n'est utilisée comme détecteur principal.
2. **Convertit** les détections en décisions de conduite via un module `VehicleState` +
   `DecisionEngine` : adaptation de vitesse, freinage d'urgence, calcul physique de la
   distance d'arrêt.
3. **Fonctionne en temps réel** sur webcam, fichier vidéo ou image fixe via OpenCV, avec un
   affichage tête haute (HUD) moderne inspiré des tableaux de bord Tesla.
4. **Reproduit l'étude de cas** de l'énoncé : un feu passant du jaune au rouge lorsque le
   véhicule approche d'une intersection, déclenchant un freinage d'urgence avec un calcul
   de distance d'arrêt.

---

##  Plan du notebook

| Section | Contenu |
|---|---|
| 1 | Installation des librairies |
| 2 | Chargement des datasets (upload manuel local) |
| 3 | Prétraitement, nettoyage et conversion au format YOLO |
| 4 | Split Train / Validation / Test + génération de `data.yaml` |
| 5 | Entraînement (YOLOv8 / YOLO11, Ultralytics) |
| 6 | Évaluation (Précision/Rappel/mAP50/mAP50-95, matrice de confusion, courbes) |
| 7 | Inférence — image / vidéo / webcam |
| 8 | Boucle de détection temps réel OpenCV + HUD |
| 9 | Logique de décision véhicule (`VehicleState`, `DecisionEngine`, distance d'arrêt) |
| 10 | Implémentation de l'étude de cas |
| 11 | Démo finale intégrée |

> **Environnement d'exécution :** ce notebook nécessite un GPU compatible CUDA (testé sur
> Google Colab, GPU Tesla T4). Les datasets sont chargés depuis des archives ZIP uploadées
> manuellement (Section 2) — aucun compte ni token Kaggle n'est requis. Toutes les cellules
> sont entièrement implémentées et s'exécutent de bout en bout, sans placeholder.

---

##  Architecture du pipeline

```
Datasets bruts (ZIP)
        │
        ▼
Prétraitement / nettoyage / conversion YOLO  (Section 3)
        │
        ▼
Split Train / Val / Test + data.yaml          (Section 4)
        │
        ▼
Entraînement YOLO11n (Ultralytics)            (Section 5)
        │
        ▼
Évaluation (mAP, matrice de confusion)        (Section 6)
        │
        ▼
Inférence temps réel (image/vidéo/webcam)     (Sections 7-8)
        │
        ▼
DecisionEngine (freinage, distance d'arrêt)   (Section 9)
        │
        ▼
Étude de cas + démo finale (HUD)              (Sections 10-11)
```

**Taxonomie unifiée des classes détectées :**

```python
CLASSES = ["speed_limit", "stop", "traffic_light_red", "traffic_light_yellow", "traffic_light_green"]
```

---

##  Datasets utilisés

Le projet combine **3 datasets image réels** (téléchargés depuis Kaggle), fusionnés en une
taxonomie unique à 5 classes pour l'entraînement du détecteur YOLO.

| Dataset | Utilisé pour | Pourquoi ce dataset |
|---|---|---|
| **GTSRB** (German Traffic Sign Recognition Benchmark) | Panneaux de limitation de vitesse (20/30/50/60/70/80/100/120 km/h), panneau Stop | Corpus de référence le plus large et le plus propre (~50 000 images, 43 classes), avec une ROI officielle par image réutilisée comme bounding box et recomposée sur des fonds de route (Section 3) pour obtenir des données d'entraînement au niveau scène. |
| **Road Sign Detection** (`andrewmvd/road-sign-detection`) | Panneaux Stop et limitation de vitesse en contexte réel de conduite | 877 images réelles, annotations bounding box au format Pascal-VOC-XML, 4 classes (`speedlimit`, `stop`, `trafficlight`, `crosswalk`). Fournit des exemples de détection **multi-objets, en contexte de scène** (contrairement aux icônes recadrées de GTSRB). Seules les classes `stop` et `speedlimit` sont retenues. |
| **Traffic Light Detection Image Set** (`farukece/traffic-light-detection-image-set`) | État du feu (rouge / jaune / vert) | Images réelles de feux de signalisation couvrant les 3 états, utilisées directement pour la détection/classification d'état — correspond exactement au besoin du projet. |

### Comment obtenir les datasets

Les datasets sont **trop volumineux pour être versionnés sur GitHub**. Ils sont fournis via
un lien Google Drive unique (3 archives ZIP prêtes à l'emploi) :

>  **Lien Google Drive (téléchargement des 3 datasets) :**
> `<< https://drive.google.com/drive/folders/1cj04_zP15IxGU9egyCREku8SqZzeNjqs?usp=sharing >>`

Après téléchargement, vous devez disposer de **3 archives ZIP** :

| Nom du fichier ZIP (doit contenir...) | Dataset | Source Kaggle (référence) |
|---|---|---|
| `gtsrb` | GTSRB | `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign` |
| `road_signs` | Road Sign Detection | `andrewmvd/road-sign-detection` |
| `traffic_lights` | Traffic Light Detection Image Set | `farukece/traffic-light-detection-image-set` |

>  Le nom exact du fichier n'a pas besoin de correspondre au caractère près — la
> Section 2.1 du notebook associe automatiquement chaque fichier uploadé au bon dataset en
> recherchant ces mots-clés dans le nom de fichier (insensible à la casse).

**Important :** conservez la structure interne de dossiers de chaque dataset telle quelle
dans le ZIP (zippez le dossier du dataset lui-même, pas seulement les fichiers en vrac) —
les Sections 3.3 à 3.5 du notebook analysent le format d'annotation natif de chaque dataset.

---

##  Installation

```bash
# Cloner le dépôt
git clone <URL_DE_VOTRE_DEPOT>
cd <nom_du_depot>

# (Recommandé) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

>  Le notebook a été conçu et testé sur **Google Colab** (GPU Tesla T4, CUDA). Il peut
> aussi tourner en local si vous disposez d'un GPU compatible CUDA ; sans GPU, l'entraînement
> fonctionnera mais sera nettement plus lent (device basculé automatiquement sur `cpu`).

---

##  Utilisation

1. **Ouvrir le notebook** `TSR_Traffic_Sign_Signal_Recognition.ipynb` sur Google Colab (ou
   Jupyter en local).
2. **Section 1** : exécuter les cellules d'installation des librairies.
3. **Section 2** : télécharger les 3 archives ZIP depuis le lien Google Drive ci-dessus, puis
   les uploader via le sélecteur de fichiers (ou les placer manuellement dans le dossier
   `manual_uploads/` indiqué par le notebook si vous n'êtes pas sur Colab).
4. **Sections 3 à 4** : exécuter le nettoyage, la conversion YOLO et le split train/val/test.
5. **Section 5** : lancer l'entraînement (modèle de base : `yolo11n.pt`, modifiable via
   `MODEL_VARIANT`).
6. **Section 6** : consulter les métriques d'évaluation (Précision, Rappel, mAP50, mAP50-95,
   matrice de confusion).
7. **Sections 7 à 11** : lancer l'inférence sur image/vidéo/webcam, la démo temps réel avec
   HUD, et l'étude de cas (transition feu jaune → rouge avec freinage d'urgence).

---

##  Logique de décision du véhicule

Le module `DecisionEngine` traduit chaque détection en action de conduite :

| Détection | Action |
|---|---|
| Panneau limitation de vitesse | Mise à jour de `target_speed_kmh` selon la limite affichée |
| Panneau Stop | Freinage d'urgence |
| Feu rouge | Freinage d'urgence |
| Feu jaune | Avertissement : « Prepare to Stop » |
| Feu vert | Poursuite de la conduite (annule tout état de freinage/avertissement) |

### Calcul physique de la distance d'arrêt

Pour un véhicule roulant à la vitesse *v* (m/s) :

- **Distance de réaction** : $d_{réaction} = v \cdot t_r$
- **Distance de freinage** : $d_{freinage} = \dfrac{v^2}{2 \mu g}$
- **Distance d'arrêt totale** : $d_{arrêt} = d_{réaction} + d_{freinage}$

Valeurs par défaut : temps de réaction $t_r = 1{,}5\,s$, coefficient de friction
$\mu = 0{,}7$ (asphalte sec), $g = 9{,}81\,m/s^2$ — les deux premiers sont configurables sur
`VehicleState`.

---

##  Résultats et évaluation

Le notebook génère automatiquement, dans le dossier `results/` :

- Les courbes de perte train/val (`results.csv` d'Ultralytics)
- La matrice de confusion et les courbes Précision/Rappel
- Un tableau récapitulatif par classe (`per_class_metrics.csv`) : Précision, Rappel, mAP50,
  mAP50-95, nombre d'images et d'instances par classe sur le jeu de test
- Une vidéo de démonstration annotée (`results/demo_result.mp4`) sur données réelles

Une vérification dédiée (« Traffic Light Class Verification ») s'assure également que le
modèle entraîné distingue bien les **3 états du feu** de façon individuelle, et n'a pas
fusionné ou perdu de classe pendant l'entraînement.

---

##  Limites connues

Le détecteur, bien qu'efficace dans les conditions représentées par ses données
d'entraînement, hérite des limites classiques des détecteurs mono-caméra / image unique
utilisés en perception pour la conduite autonome :

- **Conduite de nuit** : faible contraste, bruit capteur, éblouissement des phares
- **Pluie, brouillard, neige** : atténuation de la lumière, gouttes sur l'objectif, occlusion
- **Fort ensoleillement / éblouissement** : lavage des couleurs du feu ou du revêtement
  rétroréfléchissant des panneaux
- **Flou de mouvement / vibrations caméra** : à vitesse élevée ou sur route dégradée

---

##  Structure du dépôt

```
.
├── TSR_Traffic_Sign_Signal_Recognition.ipynb   # Notebook principal
├── requirements.txt                             # Dépendances Python
├── README.md                                    # Ce fichier
└── results/                                     # Généré à l'exécution
    ├── weights/best.pt
    ├── per_class_metrics.csv
    ├── confusion_matrix.png
    └── demo_result.mp4
```

---

##  Stack technique

- **Détection d'objets :** Ultralytics YOLOv8 / YOLO11 (PyTorch)
- **Vision par ordinateur / vidéo :** OpenCV
- **Data science :** NumPy, pandas, scikit-learn
- **Visualisation :** Matplotlib, Seaborn
- **Divers :** PyYAML, Pillow, tqdm

---

## Auteur
**Mohamed Amine Chablaoui**
**Mohamed Bouhjar**
Projet réalisé dans le cadre du module **Computer Vision — Master MIATE**.
