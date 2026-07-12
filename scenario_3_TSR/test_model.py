# TSR — Test du modèle entraîné (détection sur vidéo)
# Ce script :
#   1) Charge le modèle YOLO déjà entraîné (poids best.pt)
#   2) Exécute la détection sur une vidéo
#   3) Enregistre la vidéo annotée sur disque
#
# Installation :
#   pip install ultralytics
#
# Utilisation :
#   Placez best.pt et traffic.mp4 dans le même dossier que ce script,
#   puis lancez :
#       python test_model.py


import os
from ultralytics import YOLO

# Chemins d'entrée 
WEIGHTS_PATH = "best.pt"
VIDEO_PATH = "traffic.mp4"

# Vérification des fichiers requis
if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Fichier de poids introuvable : {WEIGHTS_PATH}\n"
        f"Placez best.pt dans le même dossier que ce script."
    )

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(
        f"Vidéo introuvable : {VIDEO_PATH}\n"
        f"Placez traffic.mp4 dans le même dossier que ce script."
    )

# Chargement du modèle 
model = YOLO(WEIGHTS_PATH)

print("Modèle chargé avec succès.")

# Prédiction sur la vidéo 
results = model.predict(
    source=VIDEO_PATH,
    save=True,
    conf=0.25
)

# Emplacement de la vidéo annotée générée 
output_path = os.path.join(results[0].save_dir, os.path.basename(VIDEO_PATH))
print(f"Vidéo annotée enregistrée dans : {output_path}")
