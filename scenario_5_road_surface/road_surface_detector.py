"""Traditional road-surface classification using visual features."""

from __future__ import annotations

import cv2
import numpy as np

from preprocessing import preprocess_road_region
from utils import SurfaceResult


def calculate_surface_features(
    processed: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Calculate color, brightness, histogram, and texture features.
    """
    gray = processed["gray"]
    hsv = processed["hsv"]
    edges = processed["edges"]

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    pixel_count = gray.size

    dark_mask = cv2.inRange(
        gray,
        0,
        95,
    )

    reflection_mask = cv2.inRange(
        gray,
        170,
        255,
    )

    bright_low_saturation_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 155], dtype=np.uint8),
        np.array([180, 75, 255], dtype=np.uint8),
    )

    very_bright_mask = cv2.inRange(
        gray,
        205,
        255,
    )

    dark_ratio = (
        cv2.countNonZero(dark_mask)
        / pixel_count
    )

    reflection_ratio = (
        cv2.countNonZero(reflection_mask)
        / pixel_count
    )

    white_ratio = (
        cv2.countNonZero(bright_low_saturation_mask)
        / pixel_count
    )

    very_bright_ratio = (
        cv2.countNonZero(very_bright_mask)
        / pixel_count
    )

    edge_ratio = (
        cv2.countNonZero(edges)
        / pixel_count
    )

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256],
    )

    histogram = histogram.flatten()
    histogram /= max(histogram.sum(), 1.0)

    dark_histogram_ratio = float(
        histogram[:96].sum()
    )

    bright_histogram_ratio = float(
        histogram[170:].sum()
    )

    return {
        "mean_brightness": float(np.mean(brightness)),
        "brightness_std": float(np.std(brightness)),
        "mean_saturation": float(np.mean(saturation)),
        "dark_ratio": float(dark_ratio),
        "reflection_ratio": float(reflection_ratio),
        "white_ratio": float(white_ratio),
        "very_bright_ratio": float(very_bright_ratio),
        "edge_ratio": float(edge_ratio),
        "dark_histogram_ratio": dark_histogram_ratio,
        "bright_histogram_ratio": bright_histogram_ratio,
    }


def classify_surface(
    features: dict[str, float],
) -> SurfaceResult:
    """
    Estimate road condition with weighted handcrafted features.

    This is a traditional computer vision classifier rather than
    a trained neural network.
    """
    dark_ratio = features["dark_ratio"]
    reflection_ratio = features["reflection_ratio"]
    white_ratio = features["white_ratio"]
    edge_ratio = features["edge_ratio"]
    brightness_std = features["brightness_std"]
    mean_brightness = features["mean_brightness"]
    mean_saturation = features["mean_saturation"]

    # Wet roads combine dark asphalt with bright reflections.
    wet_score = (
        0.40 * min(dark_ratio * 2.0, 1.0)
        + 0.35 * min(reflection_ratio * 4.5, 1.0)
        + 0.25 * min(brightness_std / 70.0, 1.0)
    )

    # Snow is usually bright, low saturation, and textured.
    snow_score = (
        0.55 * white_ratio
        + 0.25 * min(mean_brightness / 255.0, 1.0)
        + 0.20 * min(edge_ratio * 8.0, 1.0)
    )

    # Ice can appear bright and smooth with low saturation.
    ice_score = (
        0.50 * white_ratio
        + 0.25 * min(mean_brightness / 255.0, 1.0)
        + 0.25 * max(0.0, 1.0 - edge_ratio * 10.0)
    )

    if (
        snow_score >= 0.50
        and white_ratio >= 0.32
        and edge_ratio >= 0.028
    ):
        return SurfaceResult(
            condition="SNOW",
            confidence=min(snow_score, 0.99),
            features=features,
        )

    if (
        ice_score >= 0.50
        and white_ratio >= 0.25
        and mean_saturation <= 75
        and edge_ratio < 0.055
    ):
        return SurfaceResult(
            condition="ICE",
            confidence=min(ice_score, 0.99),
            features=features,
        )

    if (
        wet_score >= 0.40
        and dark_ratio >= 0.17
        and reflection_ratio >= 0.025
    ):
        return SurfaceResult(
            condition="WET",
            confidence=min(wet_score, 0.99),
            features=features,
        )

    highest_hazard_score = max(
        wet_score,
        snow_score,
        ice_score,
    )

    dry_confidence = max(
        0.50,
        1.0 - highest_hazard_score,
    )

    return SurfaceResult(
        condition="DRY",
        confidence=min(dry_confidence, 0.95),
        features=features,
    )


def detect_road_surface(
    road_region: np.ndarray,
) -> tuple[SurfaceResult, dict[str, np.ndarray]]:
    """Run preprocessing and surface classification."""
    processed = preprocess_road_region(
        road_region
    )

    features = calculate_surface_features(
        processed
    )

    result = classify_surface(features)

    return result, processed