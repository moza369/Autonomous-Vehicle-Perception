"""Image preprocessing using classical computer vision techniques."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_image(
    image: np.ndarray,
) -> np.ndarray:
    """Normalize image intensities to the range 0–255."""
    if image is None or image.size == 0:
        raise ValueError("Cannot normalize an empty image.")

    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )


def contrast_stretch(
    image: np.ndarray,
) -> np.ndarray:
    """Stretch every color channel across the full intensity range."""
    channels = cv2.split(image)
    stretched_channels = []

    for channel in channels:
        minimum = int(channel.min())
        maximum = int(channel.max())

        if maximum == minimum:
            stretched_channels.append(channel.copy())
            continue

        stretched = cv2.normalize(
            channel,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )

        stretched_channels.append(stretched)

    return cv2.merge(stretched_channels)


def preprocess_road_region(
    road_region: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Produce all intermediate images used by the detectors.
    """
    if road_region is None or road_region.size == 0:
        raise ValueError("The road region is empty.")

    normalized = normalize_image(road_region)
    stretched = contrast_stretch(normalized)

    gray = cv2.cvtColor(
        stretched,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        stretched,
        cv2.COLOR_BGR2HSV,
    )

    gaussian = cv2.GaussianBlur(
        gray,
        (7, 7),
        0,
    )

    median = cv2.medianBlur(
        gray,
        5,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_gray = clahe.apply(gaussian)

    edges = cv2.Canny(
        enhanced_gray,
        threshold1=50,
        threshold2=150,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return {
        "normalized": normalized,
        "stretched": stretched,
        "gray": gray,
        "hsv": hsv,
        "gaussian": gaussian,
        "median": median,
        "enhanced_gray": enhanced_gray,
        "edges": edges,
        "closed_edges": closed_edges,
    }