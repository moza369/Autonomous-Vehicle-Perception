"""
Scenario 8 - Detection de l'intention des vehicules
=====================================================
Detecte les feux stop, clignotants (gauche/droite) et feux de detresse
(warnings) du vehicule qui precede, a partir d'un flux video (fichier ou
webcam), et affiche le message d'avertissement/action correspondant.

Utilisation :
    python vehicle_intent_detection.py --source videos/test_final.mp4
    python vehicle_intent_detection.py --source videos/test_final.mp4 --output output/resultat.mp4
    python vehicle_intent_detection.py --source 0        # webcam

Dependances : voir requirements.txt
"""

import argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# 1. Modele et classes de vehicules
# =============================================================================

VEHICLE_CLASSES = [2, 5, 7]  # car=2, bus=5, truck=7 (classes COCO)


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    """Charge le modele YOLO pre-entraine ('n' = nano, leger et rapide)."""
    return YOLO(weights)


# =============================================================================
# 2. Detection et suivi du vehicule le plus proche
# =============================================================================

def get_closest_vehicle(results, frame_width):
    """Renvoie la boite (x1,y1,x2,y2) du vehicule le plus proche et le mieux
    centre dans l'image (probablement celui qui nous precede sur notre voie).
    Renvoie None si aucun vehicule n'est assez centre."""
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    frame_center_x = frame_width / 2
    max_offset = frame_width * 0.25  # tolerance : 25% de la largeur max

    best_score = -1
    closest_box = None

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        box_center_x = (x1 + x2) / 2
        distance_from_center = abs(box_center_x - frame_center_x)

        # On ignore completement les vehicules trop loin du centre (pas notre voie)
        if distance_from_center > max_offset:
            continue

        score = area / (1 + distance_from_center)
        if score > best_score:
            best_score = score
            closest_box = (int(x1), int(y1), int(x2), int(y2))

    return closest_box  # None si aucun vehicule n'est assez centre


def iou(box_a, box_b):
    """Intersection-over-Union entre deux boites (x1,y1,x2,y2).
    Sert a verifier qu'on continue bien de suivre LE MEME vehicule d'une
    detection YOLO a l'autre (et pas un vehicule voisin qui serait devenu
    momentanement le plus 'centre')."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


# =============================================================================
# 3. Extraction de la zone des feux et mesure d'intensite couleur
# =============================================================================

def get_taillight_region(frame, box):
    """Extrait la moitie inferieure de la boite du vehicule (zone des feux)."""
    x1, y1, x2, y2 = box
    height = y2 - y1

    light_y1 = y1 + int(height * 0.35)
    light_y2 = y2

    taillight_crop = frame[light_y1:light_y2, x1:x2]
    return taillight_crop, (x1, light_y1, x2, light_y2)


def get_red_intensity(taillight_crop):
    """Mesure le pourcentage de pixels rouges (feux stop) et leur luminosite
    moyenne dans la zone des feux."""
    hsv = cv2.cvtColor(taillight_crop, cv2.COLOR_BGR2HSV)

    # Le rouge est present a deux extremites de la teinte (Hue) en HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    red_pixel_ratio = np.sum(red_mask > 0) / red_mask.size

    if np.sum(red_mask > 0) > 0:
        avg_brightness = np.mean(hsv[:, :, 2][red_mask > 0])
    else:
        avg_brightness = 0

    return red_pixel_ratio, avg_brightness


def split_left_right(taillight_crop):
    """Coupe la zone des feux en deux moities : gauche et droite."""
    h, w = taillight_crop.shape[:2]
    mid = w // 2
    left = taillight_crop[:, :mid]
    right = taillight_crop[:, mid:]
    return left, right


def get_signal_intensity(crop):
    """Comme get_red_intensity, mais ajoute l'orange/ambre (clignotants)."""
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Rouge (feux stop / certains clignotants US)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Orange / ambre (clignotants les plus courants)
    lower_amber = np.array([11, 100, 120])
    upper_amber = np.array([35, 255, 255])

    mask = (cv2.inRange(hsv, lower_red1, upper_red1)
            | cv2.inRange(hsv, lower_red2, upper_red2)
            | cv2.inRange(hsv, lower_amber, upper_amber))

    return np.sum(mask > 0) / mask.size


# =============================================================================
# 4. Detection des clignotants et warnings (suivi ON/OFF adaptatif)
# =============================================================================

class BlinkTracker:
    """Suit l'etat ON/OFF d'un cote (gauche ou droite) et detecte le clignotement.

    - Seuil ADAPTATIF (la baseline se recalcule en continu) car l'intensite du
      rouge/ambre depend fortement de la distance du vehicule.
    - Hysteresis + debounce pour ignorer le bruit de compression.
    - Plancher de baseline (baseline_floor) : evite qu'une lecture a 0 (crop
      invalide) ne fasse chuter la baseline pres de zero et bloque le systeme.
    - Securite anti-blocage (max_on_updates) : un vrai clignotant ne reste
      jamais allume plus de ~0.6-0.7s d'affilee. Si l'etat reste bloque a ON
      plus longtemps que ca (ex: la baseline a "decroche" apres un freinage),
      on force une resynchronisation au lieu de rester fige pour le reste
      de la video.
    """

    def __init__(self, history_len=45, alpha=0.06, rel_factor=0.7, min_abs_jump=0.01,
                 min_transitions=4, debounce_frames=2, baseline_floor=0.006,
                 max_on_updates=20):
        self.history = deque(maxlen=history_len)
        self.baseline = None
        self.alpha = alpha
        self.rel_factor = rel_factor
        self.min_abs_jump = min_abs_jump
        self.min_transitions = min_transitions
        self.debounce_frames = debounce_frames
        self.baseline_floor = baseline_floor
        self.max_on_updates = max_on_updates

        self.state = False
        self._pending_state = False
        self._pending_count = 0
        self._on_updates = 0

    def update(self, ratio):
        baseline = self.baseline if self.baseline is not None else max(ratio, self.baseline_floor)
        enter_t = baseline + max(baseline * self.rel_factor, self.min_abs_jump)
        exit_t = baseline + max(baseline * self.rel_factor * 0.35, self.min_abs_jump * 0.35)

        raw = self.state
        if not self.state and ratio > enter_t:
            raw = True
        elif self.state and ratio < exit_t:
            raw = False

        if raw != self.state:
            if raw == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state = raw
                self._pending_count = 1
            if self._pending_count >= self.debounce_frames:
                self.state = raw
                self._pending_count = 0
        else:
            self._pending_count = 0

        if not self.state:
            self._on_updates = 0
            new_b = ratio if self.baseline is None else (1 - self.alpha) * self.baseline + self.alpha * ratio
            self.baseline = max(new_b, self.baseline_floor)
        else:
            self._on_updates += 1
            if self._on_updates > self.max_on_updates:
                # Anti-blocage : on force une resynchronisation
                self.baseline = max(ratio * 0.6, self.baseline_floor)
                self.state = False
                self._on_updates = 0
                self._pending_count = 0

        self.history.append(self.state)
        return self.state

    def is_blinking(self):
        if len(self.history) < 10:
            return False
        transitions = sum(
            self.history[i] != self.history[i - 1]
            for i in range(1, len(self.history))
        )
        return transitions >= self.min_transitions

    def is_steady_on(self, window=8):
        """Vrai si le cote est reste ALLUME en continu (pas de clignotement)
        -> signe d'un feu stop et non d'un clignotant."""
        if len(self.history) < window:
            return False
        recent = list(self.history)[-window:]
        return all(recent)

    def reset(self):
        """Reinitialise completement l'etat interne. INDISPENSABLE quand on
        change de vehicule suivi (ex: le "closest" bascule sur une autre
        voiture) : sans ca, le nouveau vehicule herite de la baseline et de
        l'historique de clignotement de l'ancien, ce qui fausse totalement
        la detection pendant plusieurs secondes."""
        self.history.clear()
        self.baseline = None
        self.state = False
        self._pending_state = False
        self._pending_count = 0
        self._on_updates = 0


def sync_ratio(history_a, history_b):
    """Mesure a quel point deux historiques ON/OFF sont SYNCHRONISES
    (allumes/eteints en meme temps). Les vrais warnings clignotent en phase ;
    du bruit independant sur chaque cote ne l'est pas.
    Retourne une valeur entre 0 (jamais synchro) et 1 (parfaitement synchro)."""
    n = min(len(history_a), len(history_b))
    if n == 0:
        return 0.0
    a = list(history_a)[-n:]
    b = list(history_b)[-n:]
    agree = sum(x == y for x, y in zip(a, b))
    return agree / n


# =============================================================================
# 5. Boucle principale de traitement video
# =============================================================================

def process_video(source: str, output_path: str, model: YOLO, show_debug_zones: bool = True):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source video : {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # --- Etat du freinage ---
    red_history = []
    HISTORY_SIZE = 15
    MIN_ABS_JUMP_ENTER = 0.004
    MIN_ABS_JUMP_EXIT = 0.002
    DEBOUNCE_FRAMES = 4
    braking_detected = False
    pending_state = False
    pending_count = 0

    # --- Trackers clignotants gauche / droite ---
    left_tracker = BlinkTracker()
    right_tracker = BlinkTracker()

    # Seuil de largeur RELATIF a la resolution video (portable d'une video a l'autre)
    MIN_WIDTH_RATIO = 0.10
    MIN_WIDTH_FOR_SIGNAL_SPLIT = width * MIN_WIDTH_RATIO

    # Continuite d'identite du vehicule suivi
    IOU_CONTINUITY_THRESHOLD = 0.3
    LOST_FRAMES_RESET = 10
    prev_box = None
    lost_frames = 0

    # Lissage d'affichage (evite les flashs isoles)
    MSG_STABILITY_FRAMES = 3
    last_candidate = None
    candidate_count = 0
    stable_overlay_text = None

    frame_count = 0
    closest = None

    def reset_tracking_state():
        nonlocal red_history, braking_detected, pending_state, pending_count
        red_history = []
        braking_detected = False
        pending_state = False
        pending_count = 0
        left_tracker.reset()
        right_tracker.reset()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % 3 == 0:
                results = model(frame, classes=VEHICLE_CLASSES, verbose=False)
                closest = get_closest_vehicle(results, width)

            overlay_text = None

            left_on = False
            right_on = False
            left_blinking = False
            right_blinking = False
            hazards_detected = False
            braking_final = braking_detected
            light_box = None
            x1 = y1 = x2 = y2 = None

            # Detecter un changement de vehicule / une perte de vue
            if closest is None:
                lost_frames += 1
                if lost_frames >= LOST_FRAMES_RESET:
                    reset_tracking_state()
                    prev_box = None
            else:
                lost_frames = 0
                if prev_box is not None and iou(closest, prev_box) < IOU_CONTINUITY_THRESHOLD:
                    reset_tracking_state()
                prev_box = closest

            if closest is not None:
                x1, y1, x2, y2 = closest

                taillight_crop, light_box = get_taillight_region(frame, closest)

                if taillight_crop is not None and taillight_crop.size > 0:
                    ratio, brightness = get_red_intensity(taillight_crop)

                    # --- Freinage (feux stop symetriques) ---
                    if not braking_detected:
                        red_history.append(ratio)
                        if len(red_history) > HISTORY_SIZE:
                            red_history.pop(0)

                    raw_signal = braking_detected
                    if len(red_history) >= 5:
                        baseline = np.median(red_history)
                        threshold_enter = baseline + max(baseline * 1.5, MIN_ABS_JUMP_ENTER)
                        threshold_exit = baseline + max(baseline * 0.7, MIN_ABS_JUMP_EXIT)

                        if not braking_detected and ratio > threshold_enter:
                            raw_signal = True
                        elif braking_detected and ratio < threshold_exit:
                            raw_signal = False

                    if raw_signal != braking_detected:
                        if raw_signal == pending_state:
                            pending_count += 1
                        else:
                            pending_state = raw_signal
                            pending_count = 1
                        if pending_count >= DEBOUNCE_FRAMES:
                            braking_detected = raw_signal
                            pending_count = 0
                    else:
                        pending_count = 0

                    # --- Clignotants / warnings (gauche vs droite) ---
                    vehicle_width = x2 - x1
                    if vehicle_width >= MIN_WIDTH_FOR_SIGNAL_SPLIT:
                        left_crop, right_crop = split_left_right(taillight_crop)
                        left_ratio = get_signal_intensity(left_crop)
                        right_ratio = get_signal_intensity(right_crop)
                    else:
                        left_ratio = right_ratio = 0.0

                    left_on = left_tracker.update(left_ratio)
                    right_on = right_tracker.update(right_ratio)

                    left_blinking = left_tracker.is_blinking()
                    right_blinking = right_tracker.is_blinking()

                    MIN_SYNC_RATIO = 0.75
                    sync = sync_ratio(left_tracker.history, right_tracker.history)
                    hazards_detected = left_blinking and right_blinking and sync >= MIN_SYNC_RATIO

                    left_steady = left_tracker.is_steady_on()
                    right_steady = right_tracker.is_steady_on()
                    steady_braking = left_steady and right_steady and not hazards_detected

                    braking_final = braking_detected or steady_braking

                    if braking_final and hazards_detected:
                        overlay_text = "WARNING: LEAD VEHICLE BRAKING & HAZARDS"
                    elif hazards_detected:
                        overlay_text = "WARNING: LEAD VEHICLE HAZARD LIGHTS ON"
                    elif braking_final:
                        overlay_text = "ACTION: LEAD VEHICLE BRAKING - REDUCING SPEED"
                    elif left_blinking:
                        overlay_text = "INFO: LEAD VEHICLE TURN SIGNAL - LEFT"
                    elif right_blinking:
                        overlay_text = "INFO: LEAD VEHICLE TURN SIGNAL - RIGHT"

                if hazards_detected or braking_final:
                    color = (0, 0, 255)
                elif left_blinking or right_blinking:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if show_debug_zones and light_box is not None:
                    lx1, ly1, lx2, ly2 = light_box
                    mid_x = (lx1 + lx2) // 2
                    cv2.rectangle(frame, (lx1, ly1), (mid_x, ly2),
                                  (0, 140, 255) if left_on else (100, 100, 100), 1)
                    cv2.rectangle(frame, (mid_x, ly1), (lx2, ly2),
                                  (0, 140, 255) if right_on else (100, 100, 100), 1)

            # --- Lissage d'affichage ---
            if overlay_text == last_candidate:
                candidate_count += 1
            else:
                last_candidate = overlay_text
                candidate_count = 1

            if candidate_count >= MSG_STABILITY_FRAMES:
                stable_overlay_text = overlay_text
            elif overlay_text is None:
                stable_overlay_text = None

            if stable_overlay_text:
                cv2.putText(frame, stable_overlay_text,
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            out.write(frame)
    finally:
        # S'execute TOUJOURS, meme si une erreur interrompt la boucle en plein
        # milieu, pour ne jamais laisser un fichier video mal ferme / tronque.
        cap.release()
        out.release()

    print(f"Termine ! {frame_count} frames traitees / {expected_frames} attendues, "
          f"sauvegardees dans {output_path}")
    if frame_count < expected_frames:
        print("Attention : moins de frames traitees que prevu -> la boucle a probablement "
              "ete interrompue par une erreur (verifie les logs ci-dessus).")


# =============================================================================
# 6. Point d'entree (CLI)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="videos/video_car.mp4",
                         help="Chemin de la video source, ou index de webcam (ex: 0)")
    parser.add_argument("--output", default="output/result.mp4",
                         help="Chemin du fichier video de sortie")
    parser.add_argument("--weights", default="yolov8n.pt",
                         help="Poids du modele YOLO a utiliser")
    parser.add_argument("--no-debug-zones", action="store_true",
                         help="Ne pas dessiner les zones gauche/droite analysees")
    args = parser.parse_args()

    model = load_model(args.weights)
    process_video(args.source, args.output, model, show_debug_zones=not args.no_debug_zones)


if __name__ == "__main__":
    main()
