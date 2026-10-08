"""Driver Distraction Detection Module.

Uses MediaPipe FaceMesh to detect head pitch and eye closure,
triggering cruise control disengagement after sustained inattention.
"""

import time
import cv2
import numpy as np
import mediapipe as mp

DOWN_SCORE_THRESHOLD = 0.24
EAR_THRESHOLD = 0.20
WARNING_DURATION_LIMIT = 3.0
INITIAL_SPEED = 60.0
MIN_SPEED = 30.0
SPEED_DECREASE_STEP = 0.2


def _point_to_pixel(landmark, width, height):
    return np.array([int(landmark.x * width), int(landmark.y * height)])


def _calculate_ear(face_landmarks, eye_indices, width, height):
    """Calculate Eye Aspect Ratio for one eye."""
    p1 = _point_to_pixel(face_landmarks.landmark[eye_indices["left_corner"]], width, height)
    p2 = _point_to_pixel(face_landmarks.landmark[eye_indices["right_corner"]], width, height)
    p3 = _point_to_pixel(face_landmarks.landmark[eye_indices["top_1"]], width, height)
    p4 = _point_to_pixel(face_landmarks.landmark[eye_indices["bottom_1"]], width, height)
    p5 = _point_to_pixel(face_landmarks.landmark[eye_indices["top_2"]], width, height)
    p6 = _point_to_pixel(face_landmarks.landmark[eye_indices["bottom_2"]], width, height)
    horiz = np.linalg.norm(p1 - p2)
    vert1 = np.linalg.norm(p3 - p4)
    vert2 = np.linalg.norm(p5 - p6)
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


class DriverDistractionDetector:
    """Detects driver distraction via head pitch and eye aspect ratio."""

    LEFT_EYE = {"left_corner": 33, "right_corner": 133,
                "top_1": 159, "bottom_1": 145, "top_2": 158, "bottom_2": 153}
    RIGHT_EYE = {"left_corner": 362, "right_corner": 263,
                 "top_1": 386, "bottom_1": 374, "top_2": 385, "bottom_2": 380}

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.reset()

    def process_frame(self, frame):
        """Process a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        telemetry : dict
        """
        status, down_score, ear, annotated = self._detect(frame)
        now = time.time()

        if status == "WARNING: EYES ON ROAD":
            if self._distraction_start is None:
                self._distraction_start = now
            self._distraction_duration = now - self._distraction_start
            if self._distraction_duration >= WARNING_DURATION_LIMIT:
                self._limp_mode = True
                
        else:
            self._distraction_start = None
            self._distraction_duration = 0.0
            self._limp_mode = False
            self._speed = INITIAL_SPEED

        if self._limp_mode:
            self._speed = max(MIN_SPEED, self._speed - SPEED_DECREASE_STEP)

        h, w = annotated.shape[:2]
        if self._distraction_duration > 0:
            cv2.putText(annotated, f"Distraction: {self._distraction_duration:.1f}s",
                        (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self._limp_mode:
            action_text = "ACTION: DISENGAGING CRUISE CONTROL - SLOWING DOWN"
            cv2.putText(annotated, action_text,
                        (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, f"Speed: {self._speed:.1f} km/h",
                    (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        telemetry = {
            "status": status,
            "down_score": down_score,
            "ear": ear,
            "speed": self._speed,
            "limp_mode": self._limp_mode,
            "distraction_duration": self._distraction_duration,
            "action": (
                "ACTION: DISENGAGING CRUISE CONTROL - SLOWING DOWN"
                if self._limp_mode else "NONE"
            ),
        }
        return annotated, telemetry

    def reset(self):
        self._distraction_start = None
        self._distraction_duration = 0.0
        self._limp_mode = False
        self._speed = INITIAL_SPEED

    def _detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        annotated = frame.copy()

        if not results.multi_face_landmarks:
            return "NO FACE DETECTED", None, None, annotated

        h, w, _ = frame.shape
        fl = results.multi_face_landmarks[0]

        front = fl.landmark[10]
        nose = fl.landmark[1]
        chin = fl.landmark[152]
        l_eye = fl.landmark[33]
        r_eye = fl.landmark[263]

        front_y = int(front.y * h)
        nose_y = int(nose.y * h)
        chin_y = int(chin.y * h)
        l_eye_y = int(l_eye.y * h)
        r_eye_y = int(r_eye.y * h)
        eye_center_y = (l_eye_y + r_eye_y) // 2
        face_height = chin_y - front_y

        if face_height <= 0:
            return "INVALID FACE GEOMETRY", None, None, annotated

        down_score = (nose_y - eye_center_y) / face_height
        left_ear = _calculate_ear(fl, self.LEFT_EYE, w, h)
        right_ear = _calculate_ear(fl, self.RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        head_down = down_score > DOWN_SCORE_THRESHOLD
        eyes_closed = avg_ear < EAR_THRESHOLD

        if head_down or eyes_closed:
            status, color = "WARNING: EYES ON ROAD", (0, 0, 255)
        else:
            status, color = "DRIVER ATTENTIVE", (0, 255, 0)

        pts = [
            (int(front.x * w), front_y, "front"),
            (int(nose.x * w), nose_y, "nose"),
            (int(chin.x * w), chin_y, "chin"),
            (int(l_eye.x * w), l_eye_y, "L eye"),
            (int(r_eye.x * w), r_eye_y, "R eye"),
        ]
        for x, y, label in pts:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(annotated, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(annotated, status, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated, f"Down: {down_score:.3f}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated, f"EAR: {avg_ear:.3f}",
                    (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return status, down_score, avg_ear, annotated
