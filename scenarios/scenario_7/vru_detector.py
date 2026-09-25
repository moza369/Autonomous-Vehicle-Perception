"""Vulnerable Road User (VRU) Detection Module.

Uses YOLOv8 to detect pedestrians, cyclists, and other VRUs.
Triggers autonomous emergency braking when a VRU enters the ego corridor.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class VRUDetector:
    """Detects vulnerable road users and triggers AEB."""

    PEDESTRIAN_CLASS = 0
    SPORTS_BALL_CLASS = 32
    OTHER_VRU_CLASSES = [1, 3, 15, 16, 17, 18, 21, 22]
    COOLDOWN_FRAMES = 12

    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.reset()

    def reset(self):
        self._brake_timer = 0
        self._warning_timer = 0

    def process_frame(self, frame):
        """Process a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        telemetry : dict
        """
        h, w = frame.shape[:2]
        output = frame.copy()


        trajectory_poly = np.array([
            [int(w * 0.12), int(h * 0.95)],
            [int(w * 0.42), int(h * 0.55)],
            [int(w * 0.58), int(h * 0.55)],
            [int(w * 0.88), int(h * 0.95)],
        ], np.int32)

        overlay = output.copy()
        cv2.fillPoly(overlay, [trajectory_poly], (255, 255, 0))
        cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)
        cv2.polylines(output, [trajectory_poly], True, (255, 255, 0), 2)

        results = self.model(frame, verbose=False)[0]

        pedestrian_in_lane = False
        ball_in_lane = False
        vru_count = 0
        in_corridor = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in [self.PEDESTRIAN_CLASS, self.SPORTS_BALL_CLASS] + self.OTHER_VRU_CLASSES:
                continue

            vru_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bottom_center = (int((x1 + x2) / 2), y2)
            inside = cv2.pointPolygonTest(trajectory_poly, bottom_center, False)

            if inside >= 0:
                in_corridor += 1
                if cls_id == self.PEDESTRIAN_CLASS:
                    pedestrian_in_lane = True
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.circle(output, bottom_center, 6, (0, 0, 255), -1)
                elif cls_id == self.SPORTS_BALL_CLASS:
                    ball_in_lane = True
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 165, 255), 2)
                else:
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)


        if pedestrian_in_lane:
            self._brake_timer = self.COOLDOWN_FRAMES
            self._warning_timer = 0
        elif ball_in_lane:
            if self._brake_timer == 0:
                self._warning_timer = self.COOLDOWN_FRAMES
        else:
            if self._brake_timer > 0:
                self._brake_timer -= 1
            if self._warning_timer > 0:
                self._warning_timer -= 1


        if self._brake_timer > 0:
            state = "EMERGENCY_BRAKING"
            msg = "ACTION: PEDESTRIAN DETECTED - EMERGENCY AUTOMATIC BRAKING"
            cv2.rectangle(output, (0, h - 90), (w, h), (0, 0, 255), cv2.FILLED)
            cv2.putText(output, msg, (35, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        elif self._warning_timer > 0:
            state = "WARNING"
            msg = "WARNING: OBJECT IN PATH - PRE-CHARGING BRAKE ACTUATORS"
            cv2.rectangle(output, (0, h - 90), (w, h), (0, 165, 255), cv2.FILLED)
            cv2.putText(output, msg, (35, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            state = "CRUISING"
            cv2.rectangle(output, (0, h - 90), (w, h), (30, 30, 30), cv2.FILLED)
            cv2.putText(output, "SYSTEM STATUS: PATH CLEAR - AUTONOMOUS CRUISING",
                        (35, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0), 2, cv2.LINE_AA)

        telemetry = {
            "state": state,
            "vru_count": vru_count,
            "in_corridor": in_corridor,
            "brake_timer": self._brake_timer,
            "warning_timer": self._warning_timer,
        }
        return output, telemetry
