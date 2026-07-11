# Scenario 4 Drivable Area & Free Space Detection

`The system processes individual frames from the BDD100K dataset.
Each frame is treated as a camera input, allowing the model to perform segmentation and 
decision-making without requiring a physical camera or video stream.`


NB : Ce projet implémente la détection de zone praticable (drivable area) à partir du dataset **BDD100K** et du modèle **YOLOv8-seg ONNX**.

##  Objectif de notre modèle

- Identifier les marquages de voie et les limites de route.
- Détecter les dérives du véhicule.
- Afficher une action corrective :
  - `STEERING LEFT - CORRECTING LANE DEPARTURE`
  - `STEERING RIGHT - CORRECTING LANE DEPARTURE`
  - `KEEP LANE - CENTERED`

##  Structure du projet
```text
    
    SCENARIO_4_DRIVABLE_AREA/
    ├── src/              # Code source (lane_detection.py)
    ├── models/           # Modèle ONNX (yolov8n-seg.onnx) lien drive : https://drive.google.com/drive/folders/1LkSNqhx-AFARF89v4HnmBwZzwsu2dqty?usp=drive_link
    ├── datasets/         # Dataset BDD100K , lien drive :
    ├── results/          # Exemples de sorties
    ├── requirements.txt  # Dépendances Python
    └── README.md         # Documentation
    
```

##  Installation
Pour Télécharger le modèle entraîné
Le modèle YOLOv8-seg (ONNX) n'est pas inclus dans ce repo (trop volumineux pour GitHub).

Télécharge-le ici :
[yolov8n-seg.onnx (Google Drive)](https://drive.google.com/drive/folders/1LkSNqhx-AFARF89v4HnmBwZzwsu2dqty?usp=drive_link)

```bash
pip install -r requirements.txt
```

##  Exécution

```bash
python src/lane_detection.py
```

## Le script :

    sélectionne une image aléatoire du dataset,

    applique la segmentation YOLOv8,

    calcule la dérive du véhicule,

    affiche l’action corrective sur l’image.

## Exemples de résultats

Voici quelques images générées par le modèle :

![Result 1](results/result_1.png)
![Result 2](results/result_2.png)
![Result 3](results/result_3.png)
![Result 4](results/result_4.png)
![Result 5](results/result_5.png)

## Dataset

BDD100K (inclus dans le projet)
Source officielle : https://bdd-data.berkeley.edu/

## Technologies utilisées

    Python 3.10

    OpenCV

    NumPy

    ONNX Runtime

    YOLOv8‑seg

## Case Study

    Sur une autoroute courbe, le véhicule commence à dériver vers la droite.

## Le système :

    cartographie la zone drivable et reconnaît la sortie de voie

## Puis affiche l'action :

    ACTION: STEERING LEFT - CORRECTING LANE DEPARTURE

## Auteur
`Projet réalisé par **JIMI & CHAHID ** dans le cadre du module *Computer Vision*.`
