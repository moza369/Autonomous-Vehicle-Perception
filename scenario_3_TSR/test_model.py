
# TSR — Test du modèle entraîné avec logique de décision (détection + freinage sur vidéo)

import os
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Chemins d'entrée / sortie
WEIGHTS_PATH = "best.pt"
VIDEO_PATH = "trafic5.mp4"
OUTPUT_DIR = "results"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "demo_result5.mp4")

# Vérification des fichiers requis
if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Fichier de poids introuvable : {WEIGHTS_PATH}\n"
        f"Placez best.pt dans le même dossier que ce script."
    )

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(
        f"Vidéo introuvable : {VIDEO_PATH}\n"
        f"Placez votre vidéo dans le même dossier que ce script."
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Taxonomie des classes du modèle (identique à l'entraînement)
CLASSES = ["speed_limit", "stop", "traffic_light_red", "traffic_light_yellow", "traffic_light_green"]

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

if os.environ.get("FORCE_CPU") == "1":
    DEVICE = "cpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
    print(
        "[INFO] CUDA non disponible sur cette machine (PyTorch CPU-only ou "
        "pas de GPU NVIDIA) -> exécution sur CPU. Ce sera plus lent mais "
        "fonctionnel.\n"
        "Pour utiliser le GPU, installez la version CUDA de PyTorch, ex :\n"
        "  pip uninstall torch\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )

# Palette HUD "Tesla dark-mode" (BGR) — identique au notebook
HUD_BG = (28, 24, 20)
HUD_ACCENT = (255, 255, 255)
HUD_BLUE = (255, 149, 0)
HUD_RED = (60, 60, 220)
HUD_GREEN = (110, 200, 90)
HUD_YELLOW = (30, 200, 240)

CLASS_COLORS = {
    "speed_limit": (255, 149, 0),
    "stop": (60, 60, 220),
    "traffic_light_red": (60, 60, 220),
    "traffic_light_yellow": (30, 200, 240),
    "traffic_light_green": (110, 200, 90),
}


def _overlay_panel(img, x1, y1, x2, y2, color=HUD_BG, alpha=0.55):
    """Dessine un panneau semi-transparent sur img (in-place)."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud(frame: np.ndarray, detections: List[Dict], fps: Optional[float],
             vehicle_state: Optional["VehicleState"]) -> np.ndarray:
    """Dessine les bounding boxes + le HUD (barre de statut haut, bannière d'action bas)."""
    img = frame.copy()
    h, w = img.shape[:2]

    # ---- bounding boxes ----
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_name"], HUD_BLUE)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Pour un panneau de limitation, on affiche la valeur lue par OCR
        # directement au-dessus de la boîte si elle a été détectée.
        if det["class_name"] == "speed_limit" and det.get("speed_value") is not None:
            label = f'speed_limit {det["speed_value"]:.0f} km/h {det["confidence"]*100:.0f}%'
        else:
            label = f'{det["class_name"]} {det["confidence"]*100:.0f}%'

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_ACCENT, 2, cv2.LINE_AA)

    #---barre de statut (haut)
    # Bandeau sur deux lignes : FPS sur la première, SPEED + STOPPING DIST sur
    # la seconde. Avant, tout était sur une seule ligne à des positions fixes
    # (180 px et w-tw-16 px) qui se chevauchaient dès que la vidéo était
    # étroite (portrait) ou que le texte SPEED s'allongeait -> corrigé en
    # séparant les lignes, ce qui élimine tout chevauchement quelle que soit
    # la taille de la vidéo.
    _overlay_panel(img, 0, 0, w, 90)
    fps_text = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
    cv2.putText(img, fps_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, HUD_ACCENT, 2, cv2.LINE_AA)

    if vehicle_state is not None:
        speed_text = (
            f"SPEED: {vehicle_state.current_speed_kmh:.0f} km/h  "
            f"(target {vehicle_state.target_speed_kmh:.0f})"
        )
        cv2.putText(img, speed_text, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_BLUE, 2, cv2.LINE_AA)

        dist_text = f"STOP DIST: {vehicle_state.last_stopping_distance_m:.1f} m"
        (dist_tw, _), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, dist_text, (w - dist_tw - 16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    HUD_ACCENT, 2, cv2.LINE_AA)

    # ---- bannière d'action (bas) ----
    if vehicle_state is not None and vehicle_state.current_action:
        banner_color = HUD_RED if vehicle_state.braking else HUD_BG
        _overlay_panel(img, 0, h - 70, w, h, color=banner_color, alpha=0.75)
        for i, line in enumerate(vehicle_state.current_action.split("\n")):
            cv2.putText(img, line, (16, h - 42 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        HUD_ACCENT, 2, cv2.LINE_AA)

    return img

# OCR — lecture de la valeur exacte du panneau de limitation de vitesse
# On utilise EasyOCR (déjà utilisé ailleurs dans le projet). Le lecteur est
# instancié une seule fois au démarrage (chargement du modèle OCR coûteux).
try:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
    print("Lecteur OCR (EasyOCR) chargé avec succès.")
except ImportError:
    OCR_READER = None
    print(
        "[ATTENTION] easyocr n'est pas installé -> la lecture de la valeur "
        "exacte des panneaux de limitation sera désactivée.\n"
        "Installez-le avec : pip install easyocr"
    )

# Cache pour éviter de relancer l'OCR à chaque frame sur le même panneau :
# on ne relit que toutes les N frames, et on garde la dernière valeur connue.
OCR_EVERY_N_FRAMES = 5
_last_speed_value: Optional[float] = None


def _preprocess_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Agrandit et binarise le crop du panneau pour améliorer la lecture OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Agrandissement (les petits panneaux lointains sont difficiles à lire)
    scale = 3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def read_speed_limit_value(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[float]:
    """Extrait le nombre inscrit sur un panneau de limitation de vitesse via OCR.

    Retourne la valeur en km/h (ex: 30.0, 50.0, 90.0) ou None si rien n'a pu
    être lu de façon fiable.
    """
    if OCR_READER is None:
        return None

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    processed = _preprocess_crop_for_ocr(crop)

    try:
        results = OCR_READER.readtext(processed, allowlist="0123456789")
    except Exception:
        return None

    # On garde le texte reconnu avec la plus haute confiance qui ressemble à
    # une limite de vitesse plausible (multiples de 5, entre 5 et 130 km/h).
    best_value = None
    best_conf = 0.0
    for _, text, conf in results:
        digits = re.sub(r"[^0-9]", "", text)
        if not digits:
            continue
        try:
            value = float(digits)
        except ValueError:
            continue
        if 5 <= value <= 130 and conf > best_conf:
            best_value = value
            best_conf = conf

    return best_value


# Section 9 — Logique de décision du véhicule
@dataclass
class VehicleState:
    current_speed_kmh: float = 50.0
    target_speed_kmh: float = 50.0
    reaction_time_s: float = 1.5
    friction_coeff: float = 0.7
    braking: bool = False
    warning: bool = False
    current_action: str = ""
    last_stopping_distance_m: float = 0.0
    last_detected_signal: Optional[str] = None

    def stopping_distance_m(self, speed_kmh: Optional[float] = None) -> Tuple[float, float, float]:
        """Retourne (distance de réaction, distance de freinage, distance d'arrêt) en mètres."""
        v_kmh = self.current_speed_kmh if speed_kmh is None else speed_kmh
        v_ms = v_kmh / 3.6
        g = 9.81
        d_reaction = v_ms * self.reaction_time_s
        d_braking = (v_ms ** 2) / (2 * self.friction_coeff * g) if v_ms > 0 else 0.0
        d_stopping = d_reaction + d_braking
        return d_reaction, d_braking, d_stopping

    def apply_emergency_brake(self, reason: str):
        # La distance d'arrêt affichée doit toujours répondre à la question
        # "si je freine maintenant, sur quelle distance vais-je m'arrêter ?".
        # Elle se calcule donc à partir de la vitesse ACTUELLE du véhicule au
        # moment de la détection (current_speed_kmh), jamais à partir de la
        # vitesse cible (qui vaut 0 une fois le freinage décidé).
        d_reaction, d_braking, d_stopping = self.stopping_distance_m(self.current_speed_kmh)
        self.last_stopping_distance_m = d_stopping
        self.braking = True
        self.warning = False
        self.target_speed_kmh = 0.0
        self.current_action = (
            f"ACTION:\n{reason}\nAPPLYING BRAKES  "
            f"(stopping dist: {d_stopping:.1f} m = {d_reaction:.1f} m reaction + {d_braking:.1f} m braking)"
        )
        # Le véhicule décélère immédiatement (pas de modèle physique de
        # décélération progressive dans cette démo) : la vitesse affichée
        # rejoint la cible (0) dès la frame suivante.
        self.current_speed_kmh = 0.0

    def apply_warning(self, message: str):
        self.braking = False
        self.warning = True
        self.current_action = f"WARNING:\n{message}"
        # Distance calculée à partir de la vitesse actuelle : le feu est
        # jaune, on n'a pas encore freiné, donc on montre ce qu'il faudrait
        # pour s'arrêter à la vitesse à laquelle on roule réellement.
        _, _, d_stopping = self.stopping_distance_m(self.current_speed_kmh)
        self.last_stopping_distance_m = d_stopping

    def apply_speed_update(self, new_target_kmh: float, from_ocr: bool):
        self.braking = False
        self.warning = False
        self.target_speed_kmh = new_target_kmh
        source = "OCR" if from_ocr else "DEFAUT"
        self.current_action = (
            f"ACTION:\nSPEED LIMIT DETECTED ({source})\n"
            f"TARGET SPEED SET TO {new_target_kmh:.0f} KM/H"
        )
        # Update current speed and recompute stopping distance after a new
        # speed limit is detected.
        self.current_speed_kmh = new_target_kmh
        _, _, d_stopping = self.stopping_distance_m(self.current_speed_kmh)
        self.last_stopping_distance_m = d_stopping

    def continue_driving(self):
        self.braking = False
        self.warning = False
        self.current_action = "ACTION:\nGREEN LIGHT / CLEAR\nCONTINUING"
        _, _, d_stopping = self.stopping_distance_m(self.current_speed_kmh)
        self.last_stopping_distance_m = d_stopping


DEFAULT_SPEED_LIMIT_KMH = 50.0  # valeur de repli si l'OCR ne parvient pas à lire le panneau


class DecisionEngine:
    """Transforme les détections d'une frame en mise à jour du VehicleState :
    stop/rouge -> freinage d'urgence, jaune -> avertissement,
    panneau -> vitesse cible lue par OCR, vert -> conduite normale."""

    def __init__(self, vehicle_state: VehicleState):
        self.vehicle = vehicle_state

    def _highest_confidence(self, detections: List[Dict], class_name: str) -> Optional[Dict]:
        candidates = [d for d in detections if d["class_name"] == class_name]
        return max(candidates, key=lambda d: d["confidence"]) if candidates else None

    def update(self, detections: List[Dict]) -> str:
        stop_sign = self._highest_confidence(detections, "stop")
        red_light = self._highest_confidence(detections, "traffic_light_red")
        yellow_light = self._highest_confidence(detections, "traffic_light_yellow")
        green_light = self._highest_confidence(detections, "traffic_light_green")
        speed_sign = self._highest_confidence(detections, "speed_limit")

        # Priorité : freinage d'urgence d'abord, puis avertissement, puis vitesse, puis "clear".
        if stop_sign is not None:
            self.vehicle.last_detected_signal = "stop_sign"
            self.vehicle.apply_emergency_brake("STOP SIGN DETECTED")
        elif red_light is not None:
            self.vehicle.last_detected_signal = "red_light"
            self.vehicle.apply_emergency_brake("RED LIGHT DETECTED")
        elif yellow_light is not None:
            self.vehicle.last_detected_signal = "yellow_light"
            self.vehicle.apply_warning("PREPARE TO STOP")
        elif speed_sign is not None:
            self.vehicle.last_detected_signal = "speed_limit"
            ocr_value = speed_sign.get("speed_value")
            if ocr_value is not None:
                self.vehicle.apply_speed_update(ocr_value, from_ocr=True)
            else:
                self.vehicle.apply_speed_update(DEFAULT_SPEED_LIMIT_KMH, from_ocr=False)
        elif green_light is not None:
            self.vehicle.last_detected_signal = "green_light"
            self.vehicle.continue_driving()
        else:
            # Aucune détection pertinente cette frame -> on garde l'action précédente
            # affichée, mais on recalcule la distance d'arrêt pour le HUD.
            _, _, d_stopping = self.vehicle.stopping_distance_m()
            self.vehicle.last_stopping_distance_m = d_stopping

        return self.vehicle.current_action


# Chargement du modèle

model = YOLO(WEIGHTS_PATH)
model.to(DEVICE)

print(f"Modèle chargé avec succès (device={DEVICE}).")


def detect_frame(frame_bgr: np.ndarray, frame_idx: int) -> List[Dict]:
    """Exécute le détecteur YOLO sur une frame et retourne une liste de détections
    au format {"class_name": str, "confidence": float, "box": (x1, y1, x2, y2),
    "speed_value": Optional[float]}.

    Pour les détections "speed_limit", on tente une lecture OCR de la valeur
    exacte inscrite sur le panneau (toutes les OCR_EVERY_N_FRAMES frames pour
    limiter le coût, la valeur étant réutilisée entre-temps).
    """
    global _last_speed_value

    results = model.predict(
        frame_bgr, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
        device=DEVICE, verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = CLASSES[cls_id]
        det = {
            "class_name": class_name,
            "confidence": conf,
            "box": (int(x1), int(y1), int(x2), int(y2)),
        }

        if class_name == "speed_limit":
            if frame_idx % OCR_EVERY_N_FRAMES == 0 or _last_speed_value is None:
                value = read_speed_limit_value(frame_bgr, det["box"])
                if value is not None:
                    _last_speed_value = value
            det["speed_value"] = _last_speed_value
        else:
            det["speed_value"] = None

        detections.append(det)
    return detections


# Boucle principale : détection + décision + HUD, frame par frame
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Impossible d'ouvrir la vidéo : {VIDEO_PATH}")

fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (w, h))

vehicle = VehicleState()
engine = DecisionEngine(vehicle)

prev_t = time.time()
frame_idx = 0
last_printed_action = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    detections = detect_frame(frame, frame_idx)
    engine.update(detections)

    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now

    annotated = draw_hud(frame, detections, fps=fps, vehicle_state=vehicle)
    writer.write(annotated)

    # Affiche chaque nouvelle ACTION/WARNING dans la console dès qu'elle se déclenche,
    # ex : "ACTION: RED LIGHT DETECTED | APPLYING BRAKES (...)"
    if vehicle.current_action and vehicle.current_action != last_printed_action:
        elapsed_s = frame_idx / fps_in
        flat_action = " | ".join(line for line in vehicle.current_action.split("\n") if line)
        print(f"[t={elapsed_s:6.2f}s, frame {frame_idx:5d}] {flat_action}")
        last_printed_action = vehicle.current_action

    frame_idx += 1

cap.release()
writer.release()

print(f"\nTraitement terminé : {frame_idx} frames analysées.")
print(f"Vidéo annotée enregistrée dans : {OUTPUT_PATH}")