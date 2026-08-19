#  TSR — Traffic Sign and Signal Recognition

### Système de perception et de prise de décision par Deep Learning pour la conduite autonome

**Module :** Computer Vision — Master MIATE
**Scénario :** Team Collaboration & Autonomous Vehicle Perception — Scénario 3 (TSR)

---

##  Aperçu du projet

Ce dépôt contient **deux livrables complémentaires** :

| Fichier | Rôle |
|---|---|
| `TSR_Traffic_Sign_Signal_Recognition.ipynb` | Notebook complet : préparation des datasets, entraînement YOLO11, évaluation, et démonstration temps réel intégrée. |
| `Test_model.py` | Script Python **autonome** qui recharge un modèle déjà entraîné (`best.pt`) et exécute le pipeline détection + décision sur un **fichier vidéo local**, sans dépendance à Colab/Jupyter — pratique pour un déploiement ou une démo hors notebook. |

Les deux composants partagent :

1. **La même taxonomie de classes** et le **même détecteur d'objets** entraîné (**YOLOv8 / YOLO11**, Ultralytics, PyTorch) — **aucune vision par ordinateur classique** (seuillage HSV, cercles de Hough, contours, template matching) n'est utilisée comme détecteur principal.
2. **La même logique de décision** (`VehicleState` + `DecisionEngine`) : adaptation de vitesse, freinage d'urgence, calcul physique de la distance d'arrêt.
3. Un affichage tête haute (HUD) moderne inspiré des tableaux de bord Tesla.

`Test_model.py` va plus loin que le notebook sur un point : il lit la **valeur exacte** inscrite sur un panneau de limitation de vitesse via OCR (EasyOCR), là où le notebook utilise une vitesse par défaut configurable (cf. section « Différences entre le notebook et `Test_model.py` » ci-dessous).

---

##  Plan du notebook

| Section | Contenu |
|---|---|
| 1 | Installation des librairies |
| 2 | Chargement des datasets (upload manuel local) |
| 3 | Prétraitement, nettoyage et conversion au format YOLO |
| 4 | Split Train / Validation / Test + génération de `data.yaml` |
| 5 | Entraînement (YOLO11, Ultralytics) |
| 6 | Évaluation (Précision/Rappel/mAP50/mAP50-95, matrice de confusion, courbes) |
| 7 | Inférence — image / vidéo / webcam |
| 8 | Boucle de détection temps réel OpenCV + HUD |
| 9 | Logique de décision véhicule (`VehicleState`, `DecisionEngine`, distance d'arrêt) |
| 10 | Implémentation de l'étude de cas |
| 11 | Démo finale intégrée (+ démonstration sur vidéo réelle, vérification des classes, tableau de performance par classe) |

> **Environnement d'exécution :** ce notebook nécessite un GPU compatible CUDA (testé sur
> Google Colab, GPU Tesla T4). Les datasets sont chargés depuis des archives ZIP uploadées
> manuellement (Section 2) — aucun compte ni token Kaggle n'est requis. Toutes les cellules
> sont entièrement implémentées et s'exécutent de bout en bout, sans placeholder.

---

##  `Test_model.py` — script d'inférence autonome

Une fois `best.pt` obtenu (via l'entraînement du notebook, Section 5), `Test_model.py` permet de rejouer le pipeline complet (détection + décision + HUD) sur une vidéo locale, **sans rouvrir le notebook** :

```bash
python Test_model.py
```

Par défaut, le script attend dans **son propre dossier** :

- `best.pt` — les poids du modèle entraîné,
- `trafic5.mp4` — la vidéo d'entrée (modifiable via la constante `VIDEO_PATH` en tête de script).

La vidéo annotée est écrite dans `results/demo_result.mp4`, et chaque nouvelle action (freinage, avertissement, changement de limite) est aussi affichée dans la console au moment où elle se déclenche.

**Variables d'environnement utiles :**

| Variable | Effet |
|---|---|
| `FORCE_CPU=1` | Force l'exécution sur CPU même si un GPU CUDA est disponible. |

### Emplacement des poids : notebook vs script

Le notebook entraîne et sauvegarde le modèle dans :

```
tsr_project/runs/tsr_yolo11n/weights/best.pt
```

`Test_model.py`, lui, cherche `best.pt` **directement dans le dossier où il est exécuté**. Après l'entraînement dans le notebook, copie (ou télécharge puis place) le fichier de poids à côté de `Test_model.py` :

```bash
cp tsr_project/runs/tsr_yolo11n/weights/best.pt ./best.pt
```

(Sur Colab, ce fichier peut être téléchargé directement depuis le panneau de fichiers, ou via `google.colab.files.download(...)`.)

### Différences entre le notebook et `Test_model.py`

| Aspect | Notebook (Sections 7-11) | `Test_model.py` |
|---|---|---|
| Environnement | Google Colab / Jupyter | Script Python autonome (n'importe quelle machine) |
| Source vidéo | Image / vidéo / webcam (y compris webcam navigateur en Colab) | Fichier vidéo local uniquement |
| Valeur lue sur un panneau de limitation | Valeur par défaut configurable (`DEFAULT_SPEED_LIMIT_KMH`, pas d'OCR dans le notebook) | **Lecture OCR réelle** de la valeur inscrite sur le panneau (EasyOCR), avec repli sur la valeur par défaut si l'OCR échoue ou n'est pas installé |
| Dépendance OCR | Aucune | `easyocr` (optionnelle : le script continue de fonctionner sans, avec un avertissement) |

Ces deux chemins utilisent la **même taxonomie de classes**, le **même modèle entraîné**, et la **même logique de décision** — seule la source d'entrée et la lecture (ou non) de la valeur exacte du panneau diffèrent.

---

##  Architecture du pipeline

```
Datasets bruts (ZIP)
        │
        ▼
Prétraitement / nettoyage / conversion YOLO  (Notebook, Section 3)
        │
        ▼
Split Train / Val / Test + data.yaml          (Notebook, Section 4)
        │
        ▼
Entraînement YOLO11n (Ultralytics)            (Notebook, Section 5)
        │
        ▼
Évaluation (mAP, matrice de confusion)        (Notebook, Section 6)
        │
        ▼
best.pt ─────────────┬─────────────────────────────────────────┐
                      │                                         │
                      ▼                                         ▼
        Inférence temps réel (Notebook,          Inférence sur vidéo locale
        image/vidéo/webcam, Sections 7-8)         (Test_model.py, autonome)
                      │                                         │
                      ▼                                         ▼
        DecisionEngine (freinage, distance)      DecisionEngine (identique)
        + OCR optionnel (valeur par défaut)      + OCR EasyOCR (valeur lue)
                      │                                         │
                      ▼                                         ▼
        Étude de cas + démo finale (HUD)         Vidéo annotée + log console
        (Notebook, Sections 9-11)                 (results/demo_result*.mp4)
```

**Taxonomie unifiée des classes détectées (identique dans le notebook et `Test_model.py`) :**

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

# Installer les dépendances (couvre le notebook ET Test_model.py)
pip install -r requirements.txt
```

>  Le notebook a été conçu et testé sur **Google Colab** (GPU Tesla T4, CUDA). Il peut
> aussi tourner en local si vous disposez d'un GPU compatible CUDA ; sans GPU, l'entraînement
> fonctionnera mais sera nettement plus lent (device basculé automatiquement sur `cpu`).
> `Test_model.py` suit la même règle : GPU utilisé automatiquement s'il est disponible,
> repli sur `cpu` sinon (ou forcé via `FORCE_CPU=1`).

`requirements.txt` inclut `easyocr`, utilisé uniquement par `Test_model.py` pour lire la
valeur exacte des panneaux de limitation de vitesse. Cette dépendance est optionnelle :
si elle n'est pas installée, `Test_model.py` continue de fonctionner (avertissement affiché,
repli sur la valeur de vitesse par défaut) — le notebook, lui, n'en a jamais besoin.

---

##  Utilisation

### A. Entraîner et évaluer via le notebook

1. **Ouvrir le notebook** `TSR_Traffic_Sign_Signal_Recognition.ipynb` sur Google Colab (ou
   Jupyter en local).
2. **Section 1** : exécuter les cellules d'installation des librairies.
3. **Section 2** : télécharger les 3 archives ZIP depuis le lien Google Drive ci-dessus, puis
   les uploader via le sélecteur de fichiers (ou les placer manuellement dans le dossier
   `manual_uploads/` indiqué par le notebook si vous n'êtes pas sur Colab).
4. **Sections 3 à 4** : exécuter le nettoyage, la conversion YOLO et le split train/val/test.
5. **Section 5** : lancer l'entraînement (modèle de base : `yolo11n.pt`, modifiable via
   `MODEL_VARIANT`). Les poids sont sauvegardés dans
   `tsr_project/runs/tsr_yolo11n/weights/best.pt`.
6. **Section 6** : consulter les métriques d'évaluation (Précision, Rappel, mAP50, mAP50-95,
   matrice de confusion).
7. **Sections 7 à 11** : lancer l'inférence sur image/vidéo/webcam, la démo temps réel avec
   HUD, et l'étude de cas (transition feu jaune → rouge avec freinage d'urgence).

### B. Rejouer l'inférence hors notebook avec `Test_model.py`

1. Récupérer `best.pt` depuis `tsr_project/runs/tsr_yolo11n/weights/best.pt` (notebook,
   Section 5) et le placer **dans le même dossier** que `Test_model.py`.
2. Placer votre vidéo d'entrée dans ce même dossier (nom par défaut attendu : `trafic5.mp4`,
   modifiable via `VIDEO_PATH` en tête de script).
3. Installer les dépendances si ce n'est pas déjà fait : `pip install -r requirements.txt`.
4. Lancer :
   ```bash
   python Test_model.py
   ```
5. Récupérer la vidéo annotée dans `results/demo_result5.mp4` ; suivre le log console pour
   voir chaque ACTION/WARNING déclenchée avec son horodatage.

---

##  Logique de décision du véhicule

Le module `DecisionEngine` (identique dans le notebook et `Test_model.py`) traduit chaque détection en action de conduite :

| Détection | Action |
|---|---|
| Panneau limitation de vitesse | Mise à jour de `target_speed_kmh` selon la limite affichée (lue par OCR dans `Test_model.py`, valeur par défaut dans le notebook) |
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

Le notebook génère automatiquement, dans le dossier `tsr_project/runs/` :

- Les courbes de perte train/val (`results.csv` d'Ultralytics)
- La matrice de confusion et les courbes Précision/Rappel
- Un tableau récapitulatif par classe (`per_class_metrics.csv`) : Précision, Rappel, mAP50,
  mAP50-95, nombre d'images et d'instances par classe sur le jeu de test

`Test_model.py` génère, dans le dossier `results/` (créé à côté du script) :

- Une vidéo de démonstration annotée (`results/demo_result5.mp4` par défaut)
- Un log console horodaté de chaque ACTION/WARNING déclenchée

Une vérification dédiée (« Traffic Light Class Verification », notebook) s'assure également
que le modèle entraîné distingue bien les **3 états du feu** de façon individuelle, et n'a pas
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
- **Lecture OCR (`Test_model.py` uniquement)** : la valeur exacte d'un panneau peut être mal
  lue (ou pas lue du tout) en cas de faible résolution, d'angle prononcé ou de panneau
  partiellement occulté — le repli sur `DEFAULT_SPEED_LIMIT_KMH` couvre ce cas, mais la
  vitesse cible appliquée peut alors ne pas correspondre à la limite réelle affichée.

---

##  Structure du dépôt

```
.
├── TSR_Traffic_Sign_Signal_Recognition.ipynb   # Notebook principal (entrainement + demo integree)
├── Test_model.py                                # Script d'inference autonome (video locale)
├── requirements.txt                             # Dependances Python (notebook + script)
├── README.md                                    # Ce fichier
├── best.pt                                       # A placer ici pour Test_model.py (voir Installation/Utilisation)
└── results/                                     # Genere a l'execution
    ├── demo_result.mp4                          # Sortie de Test_model.py
    └── ...                                       # Sorties du notebook (weights/, per_class_metrics.csv, confusion_matrix.png, ...)
```

---

##  Stack technique

- **Détection d'objets :** Ultralytics \ YOLO11 (PyTorch)
- **Vision par ordinateur / vidéo :** OpenCV
- **Lecture de texte sur panneaux (`Test_model.py`) :** EasyOCR (optionnel)
- **Data science :** NumPy, pandas, scikit-learn
- **Visualisation :** Matplotlib, Seaborn
- **Divers :** PyYAML, Pillow, tqdm

---

## Auteur
**Mohamed Amine Chablaoui**
**Mohamed Bouhjar**
Projet réalisé dans le cadre du module **Computer Vision — Master MIATE**.
