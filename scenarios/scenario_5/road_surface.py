"""Road Surface Condition Detection Module.

Uses classical computer vision to detect road surface conditions
(dry, wet, snow, ice) and large potholes. No deep learning required.

Adapted from Scenario 5 original implementation (5 source files merged).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SurfaceResult:
    condition: str
    confidence: float
    features: dict


@dataclass
class PotholeResult:
    boxes: list
    mask: np.ndarray

    @property
    def detected(self) -> bool:
        return len(self.boxes) > 0


@dataclass
class VehicleAction:
    warning: str
    action: str
    traction_control: str
    maximum_speed: int


def _get_road_region(frame, start_ratio=0.55):
    """Select the lower portion of the frame for analysis."""
    if frame is None or frame.size == 0:
        raise ValueError("The input frame is empty.")
    height = frame.shape[0]
    y_offset = int(height * start_ratio)
    return frame[y_offset:, :], y_offset


def _create_drivable_mask(road_region):
    """Create a trapezoidal mask approximating the drivable road area."""
    h, w = road_region.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon = np.array([[
        (int(w * 0.10), h - 1),
        (int(w * 0.36), 0),
        (int(w * 0.64), 0),
        (int(w * 0.90), h - 1),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return mask


def _preprocess(road_region):
    """Produce all intermediate images used by the detectors."""
    normalized = cv2.normalize(road_region, None, 0, 255, cv2.NORM_MINMAX)

    channels = cv2.split(normalized)
    stretched_channels = []
    for ch in channels:
        mn, mx = int(ch.min()), int(ch.max())
        if mx == mn:
            stretched_channels.append(ch.copy())
        else:
            stretched_channels.append(
                cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX))
    stretched = cv2.merge(stretched_channels)

    gray = cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(stretched, cv2.COLOR_BGR2HSV)
    gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
    median = cv2.medianBlur(gray, 5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gaussian)

    edges = cv2.Canny(enhanced_gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return {
        "gray": gray, "hsv": hsv, "gaussian": gaussian, "median": median,
        "enhanced_gray": enhanced_gray, "edges": edges,
        "closed_edges": closed_edges,
    }


def _calculate_features(processed):
    gray = processed["gray"]
    hsv = processed["hsv"]
    edges = processed["edges"]
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    pixel_count = gray.size

    dark_ratio = cv2.countNonZero(cv2.inRange(gray, 0, 95)) / pixel_count
    reflection_ratio = cv2.countNonZero(cv2.inRange(gray, 170, 255)) / pixel_count

    bls_mask = cv2.inRange(hsv,
                           np.array([0, 0, 155], np.uint8),
                           np.array([180, 75, 255], np.uint8))
    white_ratio = cv2.countNonZero(bls_mask) / pixel_count
    very_bright_ratio = cv2.countNonZero(cv2.inRange(gray, 205, 255)) / pixel_count
    edge_ratio = cv2.countNonZero(edges) / pixel_count

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist /= max(hist.sum(), 1.0)

    return {
        "mean_brightness": float(np.mean(brightness)),
        "brightness_std": float(np.std(brightness)),
        "mean_saturation": float(np.mean(saturation)),
        "dark_ratio": float(dark_ratio),
        "reflection_ratio": float(reflection_ratio),
        "white_ratio": float(white_ratio),
        "very_bright_ratio": float(very_bright_ratio),
        "edge_ratio": float(edge_ratio),
        "dark_histogram_ratio": float(hist[:96].sum()),
        "bright_histogram_ratio": float(hist[170:].sum()),
    }


def _classify_surface(features):
    dr = features["dark_ratio"]
    rr = features["reflection_ratio"]
    wr = features["white_ratio"]
    er = features["edge_ratio"]
    bs = features["brightness_std"]
    mb = features["mean_brightness"]
    ms = features["mean_saturation"]

    wet_score = (0.40 * min(dr * 2.0, 1.0)
                 + 0.35 * min(rr * 4.5, 1.0)
                 + 0.25 * min(bs / 70.0, 1.0))
    snow_score = (0.55 * wr
                  + 0.25 * min(mb / 255.0, 1.0)
                  + 0.20 * min(er * 8.0, 1.0))
    ice_score = (0.50 * wr
                 + 0.25 * min(mb / 255.0, 1.0)
                 + 0.25 * max(0.0, 1.0 - er * 10.0))

    if snow_score >= 0.50 and wr >= 0.32 and er >= 0.028:
        return SurfaceResult("SNOW", min(snow_score, 0.99), features)
    if ice_score >= 0.50 and wr >= 0.25 and ms <= 75 and er < 0.055:
        return SurfaceResult("ICE", min(ice_score, 0.99), features)
    if wet_score >= 0.40 and dr >= 0.17 and rr >= 0.025:
        return SurfaceResult("WET", min(wet_score, 0.99), features)

    highest = max(wet_score, snow_score, ice_score)
    dry_conf = max(0.50, 1.0 - highest)
    return SurfaceResult("DRY", min(dry_conf, 0.95), features)


def _detect_potholes(road_region, processed):
    enhanced_gray = processed["enhanced_gray"]
    closed_edges = processed["closed_edges"]
    h, w = enhanced_gray.shape[:2]
    roi_area = h * w

    drivable_mask = _create_drivable_mask(road_region)
    dark_mask = cv2.inRange(enhanced_gray, 0, 65)
    edge_regions = cv2.dilate(
        closed_edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    candidate = cv2.bitwise_and(dark_mask, edge_regions)
    candidate = cv2.bitwise_and(candidate, drivable_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=3)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < roi_area * 0.012 or area > roi_area * 0.12:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 70 or bh < 30:
            continue
        ar = bw / float(bh)
        if not (1.15 <= ar <= 5.5):
            continue
        cx, cy = x + bw / 2, y + bh / 2
        if cx < w * 0.15 or cx > w * 0.85 or cy < h * 0.20:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circ = 4.0 * np.pi * area / (perimeter ** 2)
        if not (0.08 <= circ <= 0.80):
            continue
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return PotholeResult(boxes=boxes[:3], mask=candidate)


def _determine_action(surface, potholes):
    cond = surface.condition.upper()
    warning = "ROAD CONDITION: NORMAL"
    action = "ACTION: NORMAL VEHICLE OPERATION"
    traction = "NORMAL"
    max_speed = 100

    if cond == "WET":
        warning, action, traction, max_speed = (
            "WARNING: WET ROAD DETECTED",
            "ACTION: REDUCING MAXIMUM SPEED", "HIGH", 60)
    elif cond == "SNOW":
        warning, action, traction, max_speed = (
            "WARNING: SNOW-COVERED ROAD DETECTED",
            "ACTION: ACTIVATING SNOW MODE", "SNOW MODE", 40)
    elif cond == "ICE":
        warning, action, traction, max_speed = (
            "WARNING: ICY ROAD DETECTED",
            "ACTION: REDUCING SPEED IMMEDIATELY", "MAXIMUM", 30)

    if potholes.detected:
        pw = ("LARGE POTHOLE DETECTED" if len(potholes.boxes) == 1
              else "MULTIPLE LARGE POTHOLES DETECTED")
        warning = f"{warning} | {pw}" if cond != "DRY" else f"WARNING: {pw}"
        action = "ACTION: SLOWING DOWN AND AVOIDING HAZARD"
        max_speed = min(max_speed, 30)

    return VehicleAction(warning, action, traction, max_speed)


DISPLAY_REGION_RATIO = 0.42


def _annotate(frame, surface, potholes, action, road_y_offset):
    output = frame.copy()
    h, w = output.shape[:2]

    dy = int(h * DISPLAY_REGION_RATIO)
    cv2.rectangle(output, (2, dy), (w - 2, h - 2), (0, 180, 0), 2)
    cv2.putText(output, "ROAD SURFACE MONITORING AREA", (8, max(dy - 6, 17)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 0), 1, cv2.LINE_AA)

    for x, y, bw, bh in potholes.boxes:
        ay = y + road_y_offset
        cv2.rectangle(output, (x, ay), (x + bw, ay + bh), (45, 45, 225), 2)
        ly = max(ay - 5, 16)
        cv2.putText(output, "LARGE POTHOLE", (x, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (55, 55, 230), 1, cv2.LINE_AA)

    panel_w = min(max(int(w * 0.38), 280), 440, w - 16)
    panel_h = min(100, h - 16)
    overlay = output.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.70, output, 0.30, 0, output)

    hazardous = surface.condition != "DRY" or potholes.detected
    wc = (65, 65, 230) if hazardous else (70, 195, 70)
    lines = [
        (action.warning, wc, 0.43),
        (action.action, (35, 150, 215), 0.38),
        (f"Traction: {action.traction_control} | Max: {action.maximum_speed} km/h",
         (200, 175, 65), 0.37),
        (f"Surface: {surface.condition} ({surface.confidence * 100:.1f}%)",
         (225, 225, 225), 0.37),
    ]
    ty = 28
    for text, color, scale in lines:
        s = scale
        while s > 0.25:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
            if tw <= panel_w - 16:
                break
            s -= 0.02
        cv2.putText(output, text, (16, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, s, color, 1, cv2.LINE_AA)
        ty += 23

    return output


class RoadSurfaceDetector:
    """Detects road surface conditions and large potholes."""

    def __init__(self):
        pass

    def process_frame(self, frame):
        """Process a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        telemetry : dict
        """
        road_region, y_offset = _get_road_region(frame)
        processed = _preprocess(road_region)
        surface = _classify_surface(_calculate_features(processed))
        potholes = _detect_potholes(road_region, processed)
        action = _determine_action(surface, potholes)
        annotated = _annotate(frame, surface, potholes, action, y_offset)

        telemetry = {
            "condition": surface.condition,
            "confidence": surface.confidence,
            "pothole_count": len(potholes.boxes),
            "warning": action.warning,
            "action": action.action,
            "traction_control": action.traction_control,
            "max_speed": action.maximum_speed,
        }
        return annotated, telemetry
