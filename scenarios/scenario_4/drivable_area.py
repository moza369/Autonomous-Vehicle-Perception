import cv2
import numpy as np


class DrivableAreaDetector:
    """Estimate the drivable corridor from lane markings.

    The bundled ``yolov8n-seg.onnx`` is a COCO object-segmentation model and
    has no drivable-area class. This detector therefore uses lane geometry and
    falls back to a centered road trapezoid when lane evidence is unavailable.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        self._last_lane_center = None

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        lane_mask, lane_center_x, left_line, right_line, confidence = (
            self._estimate_drivable_mask(frame)
        )
        self._last_lane_center = lane_center_x

        color_mask = np.zeros_like(frame)
        color_mask[lane_mask > 0] = (0, 180, 0)
        annotated_frame = cv2.addWeighted(frame, 0.72, color_mask, 0.28, 0)

        image_center_x = w // 2
        offset = lane_center_x - image_center_x
        tolerance = int(w * 0.05)

        if abs(offset) <= tolerance:
            status = "KEEP_LANE"
            action_text = "ACTION: KEEP LANE - CENTERED"
        elif offset > tolerance:
            status = "STEER_LEFT"
            action_text = "ACTION: STEERING LEFT - CORRECTING LANE DEPARTURE"
        else:
            status = "STEER_RIGHT"
            action_text = "ACTION: STEERING RIGHT - CORRECTING LANE DEPARTURE"

        if left_line is not None:
            cv2.line(
                annotated_frame,
                tuple(left_line[0]),
                tuple(left_line[1]),
                (255, 180, 0),
                3,
            )
        if right_line is not None:
            cv2.line(
                annotated_frame,
                tuple(right_line[0]),
                tuple(right_line[1]),
                (255, 180, 0),
                3,
            )
        cv2.line(annotated_frame, (image_center_x, 0),
                 (image_center_x, h), (0, 255, 0), 2)
        cv2.line(annotated_frame, (lane_center_x, 0),
                 (lane_center_x, h), (0, 0, 255), 2)
        cv2.putText(annotated_frame, action_text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Free-space confidence: {confidence:.2f}",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        telemetry = {
            "action_text": action_text,
            "status": status,
            "lane_center_x": lane_center_x,
            "image_center_x": image_center_x,
            "offset_pixels": offset,
            "confidence": confidence,
            "detection_method": "lane_geometry",
        }

        return annotated_frame, telemetry

    def _estimate_drivable_mask(self, frame):
        """Build a road corridor from lane-marking line segments."""
        height, width = frame.shape[:2]
        roi_top = int(height * 0.52)
        roi = frame[roi_top:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(
            roi_mask,
            [np.array([[
                (int(width * 0.05), roi.shape[0] - 1),
                (int(width * 0.35), int(roi.shape[0] * 0.05)),
                (int(width * 0.65), int(roi.shape[0] * 0.05)),
                (int(width * 0.95), roi.shape[0] - 1),
            ]], dtype=np.int32)],
            255,
        )
        edges = cv2.bitwise_and(edges, roi_mask)

        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(20, width // 30),
            minLineLength=max(25, width // 12),
            maxLineGap=max(15, width // 25),
        )

        left_candidates = []
        right_candidates = []
        if segments is not None:
            for segment in np.asarray(segments).reshape(-1, 4):
                x1, y1, x2, y2 = map(int, segment)
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 or abs(dy) < 10:
                    continue
                slope = dy / float(dx)
                if abs(slope) < 0.35 or abs(slope) > 5.0:
                    continue
                midpoint_x = (x1 + x2) / 2
                if slope < 0 and midpoint_x < width * 0.55:
                    left_candidates.append((x1, y1, x2, y2))
                elif slope > 0 and midpoint_x > width * 0.45:
                    right_candidates.append((x1, y1, x2, y2))

        left_line = self._fit_lane_line(left_candidates, roi_top, width, "left")
        right_line = self._fit_lane_line(right_candidates, roi_top, width, "right")

        bottom_y = height - 1
        top_y = roi_top + int(roi.shape[0] * 0.08)
        left_bottom = left_line[1][0] if left_line else int(width * 0.10)
        left_top = left_line[0][0] if left_line else int(width * 0.38)
        right_bottom = right_line[1][0] if right_line else int(width * 0.90)
        right_top = right_line[0][0] if right_line else int(width * 0.62)

        lane_polygon = np.array([[
            (left_bottom, bottom_y),
            (left_top, top_y),
            (right_top, top_y),
            (right_bottom, bottom_y),
        ]], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, lane_polygon, 255)

        lane_center_x = int((left_bottom + right_bottom) / 2)
        line_count = int(left_line is not None) + int(right_line is not None)
        confidence = 0.45 + 0.25 * line_count
        if segments is not None:
            confidence += min(len(segments), 10) * 0.02
        confidence = min(confidence, 0.95)

        return mask, lane_center_x, left_line, right_line, confidence

    @staticmethod
    def _fit_lane_line(candidates, roi_top, width, side):
        if not candidates:
            return None

        points = []
        for x1, y1, x2, y2 in candidates:
            points.extend([(x1, y1 + roi_top), (x2, y2 + roi_top)])
        xs = np.array([point[0] for point in points], dtype=np.float32)
        ys = np.array([point[1] for point in points], dtype=np.float32)

        if len(points) < 2:
            return None
        slope, intercept = np.polyfit(ys, xs, 1)
        height = max(1, int(ys.max()))
        top_y = int(max(0, ys.min()))
        bottom_x = int(np.clip(slope * height + intercept, 0, width - 1))
        top_x = int(np.clip(slope * top_y + intercept, 0, width - 1))

        if side == "left" and not top_x < width // 2:
            return None
        if side == "right" and not top_x > width // 2:
            return None
        return ((int(top_x), int(top_y)), (int(bottom_x), int(height)))
