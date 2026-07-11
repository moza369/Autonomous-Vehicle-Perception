"""Shared data structures and helper functions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SurfaceResult:
    """Road-surface classification result."""

    condition: str
    confidence: float
    features: dict[str, float]


@dataclass
class PotholeResult:
    """Large-pothole detection result."""

    boxes: list[tuple[int, int, int, int]]
    mask: np.ndarray

    @property
    def detected(self) -> bool:
        """Return True when at least one large pothole is detected."""
        return len(self.boxes) > 0


@dataclass
class VehicleAction:
    """Simulated vehicle response."""

    warning: str
    action: str
    traction_control: str
    maximum_speed: int


def get_road_region(
    frame: np.ndarray,
    start_ratio: float = 0.55,
) -> tuple[np.ndarray, int]:
    """
    Select the lower portion of the frame for actual detection.

    The detector analyzes the lower 45% of the frame by default.
    This value is kept at 0.55 because it produced more reliable
    dry/wet classification results.

    Args:
        frame: Input image or video frame.
        start_ratio: Vertical fraction where detection begins.

    Returns:
        road_region: Cropped road-analysis image.
        y_offset: Vertical position of the crop in the original frame.
    """
    if frame is None or frame.size == 0:
        raise ValueError("The input frame is empty.")

    if not 0.0 < start_ratio < 1.0:
        raise ValueError(
            "start_ratio must be between 0 and 1."
        )

    height = frame.shape[0]
    y_offset = int(height * start_ratio)

    road_region = frame[y_offset:, :]

    return road_region, y_offset


def create_drivable_mask(
    road_region: np.ndarray,
) -> np.ndarray:
    """
    Create a trapezoidal mask approximating the drivable road area.

    This reduces false pothole detections on trees, grass, sidewalks,
    shoulders, and objects near the extreme image edges.
    """
    if road_region is None or road_region.size == 0:
        raise ValueError("The road region is empty.")

    height, width = road_region.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    polygon = np.array(
        [
            [
                (int(width * 0.10), height - 1),
                (int(width * 0.36), 0),
                (int(width * 0.64), 0),
                (int(width * 0.90), height - 1),
            ]
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        polygon,
        255,
    )

    return mask


def determine_vehicle_action(
    surface: SurfaceResult,
    potholes: PotholeResult,
) -> VehicleAction:
    """
    Convert perception results into simulated vehicle decisions.
    """
    condition = surface.condition.upper()

    warning = "ROAD CONDITION: NORMAL"
    action = "ACTION: NORMAL VEHICLE OPERATION"
    traction_control = "NORMAL"
    maximum_speed = 100

    if condition == "WET":
        warning = "WARNING: WET ROAD DETECTED"
        action = "ACTION: REDUCING MAXIMUM SPEED"
        traction_control = "HIGH"
        maximum_speed = 60

    elif condition == "SNOW":
        warning = "WARNING: SNOW-COVERED ROAD DETECTED"
        action = "ACTION: ACTIVATING SNOW MODE"
        traction_control = "SNOW MODE"
        maximum_speed = 40

    elif condition == "ICE":
        warning = "WARNING: ICY ROAD DETECTED"
        action = "ACTION: REDUCING SPEED IMMEDIATELY"
        traction_control = "MAXIMUM"
        maximum_speed = 30

    if potholes.detected:
        pothole_count = len(potholes.boxes)

        if pothole_count == 1:
            pothole_warning = "LARGE POTHOLE DETECTED"
        else:
            pothole_warning = (
                "MULTIPLE LARGE POTHOLES DETECTED"
            )

        if condition == "DRY":
            warning = f"WARNING: {pothole_warning}"
        else:
            warning = (
                f"{warning} | {pothole_warning}"
            )

        action = (
            "ACTION: SLOWING DOWN AND AVOIDING HAZARD"
        )

        maximum_speed = min(
            maximum_speed,
            30,
        )

    return VehicleAction(
        warning=warning,
        action=action,
        traction_control=traction_control,
        maximum_speed=maximum_speed,
    )