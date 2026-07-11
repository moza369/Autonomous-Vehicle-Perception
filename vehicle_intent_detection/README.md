# Detection de l'intention des vehicules (Scenario 8)

Detecte automatiquement, a partir d'une video (dashcam ou webcam), l'intention
du vehicule qui precede :

- **Feux stop** (freinage)
- **Clignotant gauche** ou **droit**
- **Feux de detresse / warnings** (les deux clignotants ensemble)

Le systeme dessine une boite autour du vehicule suivi et affiche un message
d'avertissement / action a l'ecran, conformement au sujet du projet :

| Situation | Message affiche |
|---|---|
| Freinage seul | `ACTION: LEAD VEHICLE BRAKING - REDUCING SPEED` |
| Warnings seuls | `WARNING: LEAD VEHICLE HAZARD LIGHTS ON` |
| Freinage + warnings | `WARNING: LEAD VEHICLE BRAKING & HAZARDS` |
| Clignotant gauche | `INFO: LEAD VEHICLE TURN SIGNAL - LEFT` |
| Clignotant droit | `INFO: LEAD VEHICLE TURN SIGNAL - RIGHT` |

## Structure du projet

```
vehicle_intent_detection/
├── vehicle_intent_detection.py   # Script principal
├── requirements.txt              # Dependances Python
├── README.md                     # Ce fichier
├── videos/                       # Place ici tes videos source (.mp4)
└── output/                       # Les videos annotees sont generees ici
```

## Installation

```bash
pip install -r requirements.txt
```

Le modele YOLO (`yolov8n.pt`) est telecharge automatiquement au premier
lancement (connexion internet requise la premiere fois).

## Utilisation

Place ta video source dans le dossier `videos/`, puis lance :

```bash
python vehicle_intent_detection.py --source videos/test_final.mp4
```

Le resultat est enregistre dans `output/result.mp4` par defaut.

### Options disponibles

| Option | Description | Defaut |
|---|---|---|
| `--source` | Chemin de la video, ou index de webcam (`0`) | `videos/test_final.mp4` |
| `--output` | Chemin du fichier video de sortie | `output/result.mp4` |
| `--weights` | Poids YOLO a utiliser | `yolov8n.pt` |
| `--no-debug-zones` | Masque les rectangles de debug gauche/droite | (affiches par defaut) |

Exemples :

```bash
# Video personnalisee, sortie personnalisee
python vehicle_intent_detection.py --source videos/ma_video.mp4 --output output/ma_video_annotee.mp4

# Webcam en direct
python vehicle_intent_detection.py --source 0

# Sans les rectangles de debug (rendu plus propre pour une demo/rapport)
python vehicle_intent_detection.py --source videos/test_final.mp4 --no-debug-zones
```

## Comment ca marche

1. **Detection du vehicule** (`YOLOv8n`) : a chaque frame (echantillonnee 1
   fois sur 3 pour la performance), on detecte les vehicules (voiture, bus,
   camion) et on garde celui le plus proche et le mieux centre dans l'image
   (probablement notre voie de circulation).

2. **Extraction de la zone des feux** : la moitie inferieure de la boite du
   vehicule est extraite, puis coupee en deux (gauche / droite).

3. **Mesure d'intensite couleur** : conversion en HSV et mesure du
   pourcentage de pixels rouges (freinage) et rouge+ambre (clignotants) dans
   chaque zone.

4. **Detection du freinage** : comparaison a une baseline glissante (mediane
   des valeurs recentes) avec hysteresis + debounce, plus une detection
   directe pour les freinages prolonges (les deux cotes allumes en continu).

5. **Detection des clignotants/warnings** (`BlinkTracker`) : suivi ON/OFF
   adaptatif par cote, avec :
   - hysteresis (seuils differents pour allumer/eteindre) pour ignorer le
     bruit de compression video,
   - un plancher de baseline pour eviter les blocages sur des lectures a 0,
   - une securite anti-blocage si un etat reste "allume" plus longtemps
     qu'un vrai clignotant ne le ferait jamais,
   - un controle de **synchronisation** entre gauche et droite pour
     distinguer un vrai warning (les deux clignotent en phase) d'un simple
     clignotant seul.

6. **Continuite du suivi** : un calcul d'IoU (Intersection-over-Union) entre
   deux detections successives verifie qu'on suit toujours le meme vehicule.
   Si le "vehicule le plus proche" change brutalement (double-blinker,
   vehicule qui double...), tout l'etat est reinitialise pour ne pas
   melanger l'historique de deux vehicules differents.

7. **Lissage d'affichage** : un message ne s'affiche que s'il se confirme
   sur plusieurs frames consecutives, pour eviter les flashs isoles.

## Limites connues

- Quand le vehicule suivi est **trop loin** (boite trop etroite, moins de
  ~10% de la largeur de l'image), la distinction gauche/droite n'est pas
  fiable (les deux feux sont trop proches en pixels) : dans ce cas, seule la
  detection de freinage reste active.
- La detection repose sur une simple analyse de couleur (HSV), pas sur un
  modele entraine specifiquement pour les feux de vehicules : les
  performances peuvent varier selon la qualite video, la meteo et
  l'eclairage.

## Auteur

Projet realise dans le cadre du module Computer Vision (Master MIATE).
