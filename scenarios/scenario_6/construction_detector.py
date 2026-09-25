"""Construction Zone Detection Module.

Detects construction zones by identifying orange traffic cones
using HSV color segmentation. Disables lane keeping assist
and initiates merge-left maneuver.
"""

import cv2
import numpy as np


class ConstructionZoneDetector:
    """Detects construction zones via orange cone detection."""

    def __init__(self, min_cone_area=320):
        self.min_cone_area = min_cone_area
        self.orange_lower1 = np.array([0, 70, 70])
        self.orange_upper1 = np.array([25, 255, 255])
        self.orange_lower2 = np.array([160, 70, 70])
        self.orange_upper2 = np.array([180, 255, 255])

    def process_frame(self, frame):
        """Process a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        telemetry : dict
        """
        frame = cv2.resize(frame, (640, 480))
        output = frame.copy()

        cones = self._detect_cones(frame)
        merge_required = len(cones) >= 3

        for cone in cones:
            x, y, w, h = cone["box"]
            cx, cy = cone["ground_center"]
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(output, (cx, cy), 5, (0, 255, 0), -1)
            label_y = max(105, y - 5)
            cv2.putText(output, "CONE", (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        self._draw_path_line(output, cones)

        fh, fw = output.shape[:2]
        cv2.rectangle(output, (0, 0), (fw, 88), (0, 0, 0), -1)

        if merge_required:
            cv2.putText(output, "ACTION: CONSTRUCTION ZONE - MERGING LEFT",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 2)
            cv2.putText(output, "LANE KEEPING DISABLED - FOLLOWING CONES",
                        (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 255, 255), 2)
        else:
            cv2.putText(output, "NO CONSTRUCTION MERGE DETECTED",
                        (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)

        telemetry = {
            "construction_zone_detected": merge_required,
            "cone_count": len(cones),
            "action": ("CONSTRUCTION ZONE - MERGING LEFT"
                       if merge_required
                       else "NO CONSTRUCTION DETECTED"),
        }
        return output, telemetry

    def _detect_cones(self, frame):
        h_img, w_img = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.orange_lower1, self.orange_upper1)
        mask2 = cv2.inRange(hsv, self.orange_lower2, self.orange_upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask[:int(h_img * 0.18), :] = 0

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cones = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_cone_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 7 or h < 10 or w > 180 or h > 240:
                continue
            aspect_ratio = h / float(w)
            if aspect_ratio < 0.5 or aspect_ratio > 5.8:
                continue
            cx = int(x + w / 2)
            cy = int(y + h)
            cones.append({"box": (x, y, w, h),
                          "ground_center": (cx, cy), "area": area})

        cones = sorted(cones, key=lambda c: c["area"], reverse=True)[:7]
        cones = sorted(cones, key=lambda c: c["ground_center"][1])
        return cones

    def _draw_path_line(self, output_frame, cones):
        if len(cones) < 2:
            return
        points = np.array([c["ground_center"] for c in cones], dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx[0]), float(vy[0])
        x0, y0 = float(x0[0]), float(y0[0])
        slope = vx / vy if abs(vy) > 1e-6 else 0.0
        h_img, w_img = output_frame.shape[:2]
        y1 = int(np.min(points[:, 1]))
        y2 = int(np.max(points[:, 1]))
        x1 = max(0, min(w_img - 1, int(x0 + slope * (y1 - y0))))
        x2 = max(0, min(w_img - 1, int(x0 + slope * (y2 - y0))))
        cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
