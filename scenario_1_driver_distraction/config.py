# ============================================================
# Configuration du scénario 1 : Driver Distraction Detection
# ============================================================

# Index de la webcam utilisée par OpenCV
CAMERA_INDEX = 0

# Seuil de détection de tête inclinée vers le bas
DOWN_SCORE_THRESHOLD = 0.24

# Seuil de Eye Aspect Ratio
# Si EAR est inférieur à cette valeur, les yeux sont considérés
# comme trop fermés ou non dirigés vers la route.
EAR_THRESHOLD = 0.20

# Durée maximale de distraction avant action automatique
WARNING_DURATION_LIMIT = 3.0

# Vitesse simulée du véhicule en km/h
INITIAL_SPEED = 60.0

# Vitesse minimale après ralentissement
MIN_SPEED = 30.0

# Diminution progressive de la vitesse à chaque frame
SPEED_DECREASE_STEP = 0.2
