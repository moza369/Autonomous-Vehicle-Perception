"""Large-pothole detection using edges, morphology, and contours."""

from __future__ import annotations

import cv2
import numpy as np

from utils import (
    PotholeResult,
    create_drivable_mask,
)


def detect_large_potholes(
    road_region: np.ndarray,
    processed: dict[str, np.ndarray],
) -> PotholeResult:
    """
    Detect large dark and irregular pothole-like regions.

    The size threshold intentionally ignores small road defects that
    are unlikely to affect driving.
    """
    enhanced_gray = processed["enhanced_gray"]
    closed_edges = processed["closed_edges"]

    height, width = enhanced_gray.shape[:2]
    roi_area = height * width

    drivable_mask = create_drivable_mask(
        road_region
    )

    # Select dark depressions.
    dark_mask = cv2.inRange(
        enhanced_gray,
        0,
        65,
    )

    # Combine darkness with structural edges.
    edge_regions = cv2.dilate(
        closed_edges,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
        iterations=2,
    )

    candidate_mask = cv2.bitwise_and(
        dark_mask,
        edge_regions,
    )

    candidate_mask = cv2.bitwise_and(
        candidate_mask,
        drivable_mask,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3,
    )

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Large potholes only:
        # at least 1.2% of the visible road-region area.
        if area < roi_area * 0.012:
            continue

        # Ignore regions that are unrealistically large.
        if area > roi_area * 0.12:
            continue

        x, y, box_width, box_height = cv2.boundingRect(
            contour
        )

        if box_width < 70 or box_height < 30:
            continue

        aspect_ratio = (
            box_width / float(box_height)
        )

        # Potholes generally appear wider than tall.
        if not 1.15 <= aspect_ratio <= 5.5:
            continue

        center_x = x + box_width / 2
        center_y = y + box_height / 2

        # Ignore extreme image edges.
        if center_x < width * 0.15:
            continue

        if center_x > width * 0.85:
            continue

        # Prefer hazards closer to the vehicle.
        if center_y < height * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            * np.pi
            * area
            / (perimeter * perimeter)
        )

        if not 0.08 <= circularity <= 0.80:
            continue

        boxes.append(
            (x, y, box_width, box_height)
        )

    boxes.sort(
        key=lambda box: box[2] * box[3],
        reverse=True,
    )

    # Display at most the three largest driving hazards.
    return PotholeResult(
        boxes=boxes[:3],
        mask=candidate_mask,
    )