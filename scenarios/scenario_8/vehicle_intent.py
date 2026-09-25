"""Vehicle Intent & Tail Light Detection Module.

Tracks the lead vehicle's tail lights to detect braking, turn signals,
and hazard lights using HSV color analysis and temporal blink tracking.

Adapted from Scenario 8 original implementation.
"""

from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck (COCO)


def _get_closest_vehicle(results, frame_width):
    """Return the bounding box of the closest centred vehicle."""
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    center_x = frame_width / 2
    max_offset = frame_width * 0.25
    best_score, closest = -1, None
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        dx = abs((x1 + x2) / 2 - center_x)
        if dx > max_offset:
            continue
        score = area / (1 + dx)
        if score > best_score:
            best_score = score
            closest = (int(x1), int(y1), int(x2), int(y2))
    return closest


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _get_taillight_region(frame, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    ly1 = y1 + int(h * 0.35)
    crop = frame[ly1:y2, x1:x2]
    return crop, (x1, ly1, x2, y2)


def _get_red_intensity(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = m1 + m2
    ratio = np.sum(red_mask > 0) / red_mask.size
    brightness = float(np.mean(hsv[:, :, 2][red_mask > 0])) if np.sum(red_mask > 0) > 0 else 0
    return ratio, brightness


def _split_left_right(crop):
    h, w = crop.shape[:2]
    mid = w // 2
    return crop[:, :mid], crop[:, mid:]


def _get_signal_intensity(crop):
    """Return the proportion of low-light amber indicator pixels."""
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Indicators are commonly amber. Keep red excluded so brake lights
    # cannot masquerade as a turn signal, and allow dark-frame values.
    mask = cv2.inRange(
        hsv,
        np.array([8, 45, 25]),
        np.array([40, 255, 255]),
    )
    return np.sum(mask > 0) / mask.size


def _sync_ratio(hist_a, hist_b):
    n = min(len(hist_a), len(hist_b))
    if n == 0:
        return 0.0
    a, b = list(hist_a)[-n:], list(hist_b)[-n:]
    return sum(x == y for x, y in zip(a, b)) / n


class _BlinkTracker:
    def __init__(self, history_len=45, alpha=0.06, rel_factor=0.7,
                 min_abs_jump=0.003, min_transitions=3, debounce_frames=2,
                 baseline_floor=0.002, max_on_updates=20):
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
        bl = self.baseline if self.baseline is not None else max(ratio, self.baseline_floor)
        enter_t = bl + max(bl * self.rel_factor, self.min_abs_jump)
        exit_t = bl + max(bl * self.rel_factor * 0.35, self.min_abs_jump * 0.35)

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
                self.baseline = max(ratio * 0.6, self.baseline_floor)
                self.state = False
                self._on_updates = 0
                self._pending_count = 0

        self.history.append(self.state)
        return self.state

    def is_blinking(self):
        if len(self.history) < 10:
            return False
        return sum(self.history[i] != self.history[i - 1]
                   for i in range(1, len(self.history))) >= self.min_transitions

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


class VehicleIntentDetector:
    """Detects lead vehicle braking, turn signals, and hazard lights."""

    HISTORY_SIZE = 15
    MIN_ABS_JUMP_ENTER = 0.004
    MIN_ABS_JUMP_EXIT = 0.002
    DEBOUNCE_FRAMES = 4
    IOU_THRESHOLD = 0.3
    LOST_RESET = 10
    MSG_STABILITY = 3
    MIN_WIDTH_RATIO = 0.10
    MIN_INDICATOR_RATIO = 0.008
    SIDE_DOMINANCE_RATIO = 1.20

    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.reset()

    def reset(self):
        self._red_history = []
        self._braking = False
        self._pending_state = False
        self._pending_count = 0
        self._left_tracker = _BlinkTracker()
        self._right_tracker = _BlinkTracker()
        self._prev_box = None
        self._lost_frames = 0
        self._last_candidate = None
        self._candidate_count = 0
        self._stable_text = None
        self._frame_count = 0
        self._closest = None

    def _reset_tracking(self):
        self._red_history = []
        self._braking = False
        self._pending_state = False
        self._pending_count = 0
        self._left_tracker.reset()
        self._right_tracker.reset()

    def process_frame(self, frame):
        """Process a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        telemetry : dict
        """
        h, w = frame.shape[:2]
        output = frame.copy()
        self._frame_count += 1

        # Run YOLO every 3 frames
        if self._frame_count % 3 == 0:
            results = self.model(frame, classes=VEHICLE_CLASSES, verbose=False)
            self._closest = _get_closest_vehicle(results, w)

        overlay_text = None
        left_blinking = right_blinking = hazards = False
        braking_final = self._braking
        intent = "NONE"

        # Identity continuity check
        if self._closest is None:
            self._lost_frames += 1
            if self._lost_frames >= self.LOST_RESET:
                self._reset_tracking()
                self._prev_box = None
        else:
            self._lost_frames = 0
            if self._prev_box is not None and _iou(self._closest, self._prev_box) < self.IOU_THRESHOLD:
                self._reset_tracking()
            self._prev_box = self._closest

        if self._closest is not None:
            x1, y1, x2, y2 = self._closest
            crop, light_box = _get_taillight_region(frame, self._closest)

            if crop is not None and crop.size > 0:
                ratio, brightness = _get_red_intensity(crop)

                # Braking detection
                if not self._braking:
                    self._red_history.append(ratio)
                    if len(self._red_history) > self.HISTORY_SIZE:
                        self._red_history.pop(0)

                raw_signal = self._braking
                if len(self._red_history) >= 5:
                    baseline = np.median(self._red_history)
                    t_enter = baseline + max(baseline * 1.5, self.MIN_ABS_JUMP_ENTER)
                    t_exit = baseline + max(baseline * 0.7, self.MIN_ABS_JUMP_EXIT)
                    if not self._braking and ratio > t_enter:
                        raw_signal = True
                    elif self._braking and ratio < t_exit:
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

                # Turn signal / hazard detection
                vw = x2 - x1
                min_w = w * self.MIN_WIDTH_RATIO
                if vw >= min_w:
                    lc, rc = _split_left_right(crop)
                    lr, rr = _get_signal_intensity(lc), _get_signal_intensity(rc)
                else:
                    lr = rr = 0.0

                self._left_tracker.update(lr)
                self._right_tracker.update(rr)
                left_blinking = self._left_tracker.is_blinking()
                right_blinking = self._right_tracker.is_blinking()

                # If processing starts while a signal is already illuminated,
                # there may not be enough OFF/ON transitions for the blink
                # tracker yet. Use amber side dominance as an immediate,
                # low-light fallback until temporal evidence is available.
                left_indicator = (
                    lr >= self.MIN_INDICATOR_RATIO
                    and lr > rr * self.SIDE_DOMINANCE_RATIO
                )
                right_indicator = (
                    rr >= self.MIN_INDICATOR_RATIO
                    and rr > lr * self.SIDE_DOMINANCE_RATIO
                )
                left_blinking = left_blinking or left_indicator
                right_blinking = right_blinking or right_indicator

                sync = _sync_ratio(self._left_tracker.history, self._right_tracker.history)
                hazards = left_blinking and right_blinking and sync >= 0.75

                steady_braking = (self._left_tracker.is_steady_on()
                                  and self._right_tracker.is_steady_on()
                                  and not hazards)
                braking_final = self._braking or steady_braking

                if braking_final and hazards:
                    overlay_text = "WARNING: LEAD VEHICLE BRAKING & HAZARDS"
                    intent = "BRAKING_HAZARDS"
                elif hazards:
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

            # Draw vehicle box
            if hazards or braking_final:
                color = (0, 0, 255)
            elif left_blinking or right_blinking:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Debug zones
            if light_box is not None:
                lx1, ly1, lx2, ly2 = light_box
                mid_x = (lx1 + lx2) // 2
                lc = (0, 140, 255) if self._left_tracker.state else (100, 100, 100)
                rc = (0, 140, 255) if self._right_tracker.state else (100, 100, 100)
                cv2.rectangle(output, (lx1, ly1), (mid_x, ly2), lc, 1)
                cv2.rectangle(output, (mid_x, ly1), (lx2, ly2), rc, 1)

        # Message stability filter
        if overlay_text == self._last_candidate:
            self._candidate_count += 1
        else:
            self._last_candidate = overlay_text
            self._candidate_count = 1

        if self._candidate_count >= self.MSG_STABILITY:
            self._stable_text = overlay_text
        elif overlay_text is None:
            self._stable_text = None

        if self._stable_text:
            cv2.putText(output, self._stable_text,
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        telemetry = {
            "status_message": self._stable_text or "MONITORING LEAD VEHICLE",
            "intent": intent,
            "brake_detected": braking_final,
            "left_blinking": left_blinking,
            "right_blinking": right_blinking,
            "hazards": hazards,
        }
        return output, telemetry
