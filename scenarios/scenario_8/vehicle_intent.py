"""Vehicle Intent & Tail Light Detection Module (Scenario 8).

Detects the brake lights, turn signals (left/right) and hazard lights of the
lead vehicle.
"""

import os
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASSES = [2, 5, 7]  # car=2, bus=5, truck=7 (COCO)


# =============================================================================
# 1. Vehicle selection and identity continuity
# =============================================================================

def get_closest_vehicle(results, frame_width):
    """Return the box (x1,y1,x2,y2) of the closest and most centred vehicle."""
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    frame_center_x = frame_width / 2
    max_offset = frame_width * 0.25

    best_score = -1
    closest_box = None

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        box_center_x = (x1 + x2) / 2
        distance_from_center = abs(box_center_x - frame_center_x)

        if distance_from_center > max_offset:
            continue

        score = area / (1 + distance_from_center)
        if score > best_score:
            best_score = score
            closest_box = (int(x1), int(y1), int(x2), int(y2))

    return closest_box


def iou(box_a, box_b):
    """Intersection-over-Union between two boxes (x1,y1,x2,y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


# =============================================================================
# 2. Tail light region and colour intensity
# =============================================================================

def get_taillight_region(frame, box):
    """Lower part of the vehicle box (tail light area)."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    light_y1 = y1 + int(height * 0.35)
    light_y2 = y2
    crop = frame[light_y1:light_y2, x1:x2]
    return crop, (x1, light_y1, x2, light_y2)


def get_red_intensity(taillight_crop):
    """Percentage of red pixels (brake lights) and their average brightness."""
    hsv = cv2.cvtColor(taillight_crop, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = mask1 + mask2

    red_pixel_ratio = np.sum(red_mask > 0) / red_mask.size

    if np.sum(red_mask > 0) > 0:
        avg_brightness = np.mean(hsv[:, :, 2][red_mask > 0])
    else:
        avg_brightness = 0

    return red_pixel_ratio, avg_brightness


def split_left_right(taillight_crop):
    """Split the tail light area into a left half and a right half."""
    h, w = taillight_crop.shape[:2]
    mid = w // 2
    return taillight_crop[:, :mid], taillight_crop[:, mid:]


def get_signal_intensity(crop):
    """Percentage of red + orange/amber pixels (turn signals)."""
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_amber = np.array([11, 100, 120])
    upper_amber = np.array([35, 255, 255])

    mask = (cv2.inRange(hsv, lower_red1, upper_red1)
            | cv2.inRange(hsv, lower_red2, upper_red2)
            | cv2.inRange(hsv, lower_amber, upper_amber))

    return np.sum(mask > 0) / mask.size


# =============================================================================
# 3. Blink tracking
# =============================================================================

class BlinkTracker:
    """Track the ON/OFF state of one side (left or right) and detect blinking
    (adaptive threshold, hysteresis, debounce, baseline floor, anti-lock)."""

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
                # Anti-lock: force a resynchronisation
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
        if len(self.history) < window:
            return False
        return all(list(self.history)[-window:])

    def reset(self):
        self.history.clear()
        self.baseline = None
        self.state = False
        self._pending_state = False
        self._pending_count = 0
        self._on_updates = 0


def sync_ratio(history_a, history_b):
    """Synchronisation (0..1) between two ON/OFF histories."""
    n = min(len(history_a), len(history_b))
    if n == 0:
        return 0.0
    a = list(history_a)[-n:]
    b = list(history_b)[-n:]
    return sum(x == y for x, y in zip(a, b)) / n


# =============================================================================
# 4. Integration class (interface required by the main project)
# =============================================================================

class VehicleIntentDetector:
    """Detects braking, turn signals and hazard lights of the lead vehicle."""

    # --- Braking ---
    HISTORY_SIZE = 15
    MIN_ABS_JUMP_ENTER = 0.004
    MIN_ABS_JUMP_EXIT = 0.002
    DEBOUNCE_FRAMES = 4

    # --- Tracked vehicle ---
    MIN_WIDTH_RATIO = 0.10
    IOU_CONTINUITY_THRESHOLD = 0.3
    LOST_FRAMES_RESET = 10
    YOLO_EVERY_N_FRAMES = 3

    # --- Hazards / display ---
    MIN_SYNC_RATIO = 0.75
    MSG_STABILITY_FRAMES = 3

    def __init__(self, model_path="models/yolov8n.pt"):
        # If the weights file does not exist, let ultralytics download it
        # (short name "yolov8n.pt") instead of crashing.
        if model_path and not os.path.exists(model_path):
            model_path = os.path.basename(model_path) or "yolov8n.pt"
        self.model = YOLO(model_path)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Full reset (new video / new scenario)."""
        self._red_history = []
        self._braking = False
        self._pending_state = False
        self._pending_count = 0

        self._left_tracker = BlinkTracker()
        self._right_tracker = BlinkTracker()

        self._prev_box = None
        self._lost_frames = 0

        self._last_candidate = None
        self._candidate_count = 0
        self._stable_text = None

        self._frame_count = 0
        self._closest = None

    def _reset_tracking(self):
        """Reset when the tracked vehicle changes (keeps global counters
        and the display state)."""
        self._red_history = []
        self._braking = False
        self._pending_state = False
        self._pending_count = 0
        self._left_tracker.reset()
        self._right_tracker.reset()

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Process ONE BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        telemetry : dict
        """
        if frame is None:
            return frame, {
                "status_message": "MONITORING LEAD VEHICLE",
                "intent": "NONE",
                "brake_detected": False,
                "left_blinking": False,
                "right_blinking": False,
                "hazards": False,
            }

        height, width = frame.shape[:2]
        annotated = frame.copy()
        self._frame_count += 1

        # Run YOLO once every 3 frames
        if self._frame_count % self.YOLO_EVERY_N_FRAMES == 0:
            results = self.model(frame, classes=VEHICLE_CLASSES, verbose=False)
            self._closest = get_closest_vehicle(results, width)

        closest = self._closest
        overlay_text = None
        intent = "NONE"

        left_on = right_on = False
        left_blinking = right_blinking = False
        hazards_detected = False
        braking_final = self._braking
        light_box = None

        # --- Identity continuity of the tracked vehicle ---
        if closest is None:
            self._lost_frames += 1
            if self._lost_frames >= self.LOST_FRAMES_RESET:
                self._reset_tracking()
                self._prev_box = None
        else:
            self._lost_frames = 0
            if self._prev_box is not None and iou(closest, self._prev_box) < self.IOU_CONTINUITY_THRESHOLD:
                self._reset_tracking()
            self._prev_box = closest

        if closest is not None:
            x1, y1, x2, y2 = closest
            taillight_crop, light_box = get_taillight_region(frame, closest)

            if taillight_crop is not None and taillight_crop.size > 0:
                ratio, _brightness = get_red_intensity(taillight_crop)

                # --- Braking (symmetrical brake lights) ---
                if not self._braking:
                    self._red_history.append(ratio)
                    if len(self._red_history) > self.HISTORY_SIZE:
                        self._red_history.pop(0)

                raw_signal = self._braking
                if len(self._red_history) >= 5:
                    baseline = np.median(self._red_history)
                    threshold_enter = baseline + max(baseline * 1.5, self.MIN_ABS_JUMP_ENTER)
                    threshold_exit = baseline + max(baseline * 0.7, self.MIN_ABS_JUMP_EXIT)

                    if not self._braking and ratio > threshold_enter:
                        raw_signal = True
                    elif self._braking and ratio < threshold_exit:
                        raw_signal = False

                if raw_signal != self._braking:
                    if raw_signal == self._pending_state:
                        self._pending_count += 1
                    else:
                        self._pending_state = raw_signal
                        self._pending_count = 1
                    if self._pending_count >= self.DEBOUNCE_FRAMES:
                        self._braking = raw_signal
                        self._pending_count = 0
                else:
                    self._pending_count = 0

                # --- Turn signals / hazards (left vs right) ---
                vehicle_width = x2 - x1
                if vehicle_width >= width * self.MIN_WIDTH_RATIO:
                    left_crop, right_crop = split_left_right(taillight_crop)
                    left_ratio = get_signal_intensity(left_crop)
                    right_ratio = get_signal_intensity(right_crop)
                else:
                    left_ratio = right_ratio = 0.0

                left_on = self._left_tracker.update(left_ratio)
                right_on = self._right_tracker.update(right_ratio)

                left_blinking = self._left_tracker.is_blinking()
                right_blinking = self._right_tracker.is_blinking()

                sync = sync_ratio(self._left_tracker.history, self._right_tracker.history)
                hazards_detected = left_blinking and right_blinking and sync >= self.MIN_SYNC_RATIO

                steady_braking = (self._left_tracker.is_steady_on()
                                  and self._right_tracker.is_steady_on()
                                  and not hazards_detected)

                braking_final = self._braking or steady_braking

                if braking_final and hazards_detected:
                    overlay_text = "WARNING: LEAD VEHICLE BRAKING & HAZARDS"
                    intent = "BRAKING_HAZARDS"
                elif hazards_detected:
                    overlay_text = "WARNING: LEAD VEHICLE HAZARD LIGHTS ON"
                    intent = "HAZARDS"
                elif braking_final:
                    overlay_text = "ACTION: LEAD VEHICLE BRAKING - REDUCING SPEED"
                    intent = "BRAKING"
                elif left_blinking:
                    overlay_text = "INFO: LEAD VEHICLE TURN SIGNAL - LEFT"
                    intent = "TURN_LEFT"
                elif right_blinking:
                    overlay_text = "INFO: LEAD VEHICLE TURN SIGNAL - RIGHT"
                    intent = "TURN_RIGHT"

            # --- Draw the vehicle box ---
            if hazards_detected or braking_final:
                color = (0, 0, 255)
            elif left_blinking or right_blinking:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # --- Analysed left / right zones ---
            if light_box is not None:
                lx1, ly1, lx2, ly2 = light_box
                mid_x = (lx1 + lx2) // 2
                cv2.rectangle(annotated, (lx1, ly1), (mid_x, ly2),
                              (0, 140, 255) if left_on else (100, 100, 100), 1)
                cv2.rectangle(annotated, (mid_x, ly1), (lx2, ly2),
                              (0, 140, 255) if right_on else (100, 100, 100), 1)

        # --- Display smoothing ---
        if overlay_text == self._last_candidate:
            self._candidate_count += 1
        else:
            self._last_candidate = overlay_text
            self._candidate_count = 1

        if self._candidate_count >= self.MSG_STABILITY_FRAMES:
            self._stable_text = overlay_text
        elif overlay_text is None:
            self._stable_text = None

        if self._stable_text:
            cv2.putText(annotated, self._stable_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        telemetry = {
            "status_message": self._stable_text or "MONITORING LEAD VEHICLE",
            "intent": intent,
            "brake_detected": bool(braking_final),
            "left_blinking": bool(left_blinking),
            "right_blinking": bool(right_blinking),
            "hazards": bool(hazards_detected),
        }
        return annotated, telemetry
