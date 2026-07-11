"""Compact visual overlays for road-condition warnings and actions."""

from __future__ import annotations

import cv2
import numpy as np

from utils import (
    PotholeResult,
    SurfaceResult,
    VehicleAction,
)


# The visible green monitoring area starts higher than the real
# classification crop. Changing this does not affect classification.
DISPLAY_REGION_RATIO = 0.42


def calculate_font_scale(
    text: str,
    maximum_width: int,
    preferred_scale: float,
    minimum_scale: float = 0.25,
) -> float:
    """
    Reduce font size until text fits inside the compact status panel.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = preferred_scale

    while scale > minimum_scale:
        (text_width, _), _ = cv2.getTextSize(
            text,
            font,
            scale,
            1,
        )

        if text_width <= maximum_width:
            return scale

        scale -= 0.02

    return minimum_scale


def draw_status_panel(
    frame: np.ndarray,
    surface: SurfaceResult,
    potholes: PotholeResult,
    action: VehicleAction,
) -> None:
    """
    Draw a small semi-transparent panel in the upper-left corner.
    """
    frame_height, frame_width = frame.shape[:2]

    panel_x = 8
    panel_y = 8

    panel_width = min(
        max(int(frame_width * 0.38), 280),
        440,
        frame_width - 16,
    )

    panel_height = min(
        100,
        frame_height - 16,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (12, 12, 12),
        -1,
    )

    # Semi-transparent panel preserves visibility behind the text.
    cv2.addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0,
        frame,
    )

    hazardous = (
        surface.condition != "DRY"
        or potholes.detected
    )

    warning_color = (
        (65, 65, 230)
        if hazardous
        else (70, 195, 70)
    )

    lines = [
        (
            action.warning,
            warning_color,
            0.43,
        ),
        (
            action.action,
            (35, 150, 215),
            0.38,
        ),
        (
            (
                f"Traction: {action.traction_control}"
                f" | Max: {action.maximum_speed} km/h"
            ),
            (200, 175, 65),
            0.37,
        ),
        (
            (
                f"Surface: {surface.condition}"
                f" ({surface.confidence * 100:.1f}%)"
            ),
            (225, 225, 225),
            0.37,
        ),
    ]

    text_x = panel_x + 8
    text_y = panel_y + 20
    usable_width = panel_width - 16

    for text, color, preferred_scale in lines:
        font_scale = calculate_font_scale(
            text=text,
            maximum_width=usable_width,
            preferred_scale=preferred_scale,
        )

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

        text_y += 23


def draw_visual_monitoring_region(
    frame: np.ndarray,
) -> None:
    """
    Draw a green monitoring outline beginning above the image middle.

    This is only a display overlay. The real detector still uses the
    lower 45% crop defined in utils.get_road_region().
    """
    height, width = frame.shape[:2]

    display_y_offset = int(
        height * DISPLAY_REGION_RATIO
    )

    cv2.rectangle(
        frame,
        (2, display_y_offset),
        (width - 2, height - 2),
        (0, 180, 0),
        2,
    )

    label = "ROAD SURFACE MONITORING AREA"

    label_x = 8
    label_y = max(
        display_y_offset - 6,
        17,
    )

    # Dark outline behind the green text.
    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 210, 0),
        1,
        cv2.LINE_AA,
    )


def draw_potholes(
    frame: np.ndarray,
    potholes: PotholeResult,
    detection_y_offset: int,
) -> None:
    """
    Draw compact bounding boxes around detected large potholes.

    Pothole coordinates use the real detection crop, so the original
    detection_y_offset must still be used here.
    """
    for x, y, width, height in potholes.boxes:
        adjusted_y = y + detection_y_offset

        cv2.rectangle(
            frame,
            (x, adjusted_y),
            (
                x + width,
                adjusted_y + height,
            ),
            (45, 45, 225),
            2,
        )

        label_y = max(
            adjusted_y - 5,
            16,
        )

        # Dark outline for readability.
        cv2.putText(
            frame,
            "LARGE POTHOLE",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "LARGE POTHOLE",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (55, 55, 230),
            1,
            cv2.LINE_AA,
        )


def draw_debug_information(
    frame: np.ndarray,
    surface: SurfaceResult,
    potholes: PotholeResult,
) -> None:
    """
    Draw compact debug values in the upper-right corner.
    """
    features = surface.features

    debug_lines = [
        f"Dark: {features.get('dark_ratio', 0.0):.2f}",
        (
            "Reflection: "
            f"{features.get('reflection_ratio', 0.0):.2f}"
        ),
        f"White: {features.get('white_ratio', 0.0):.2f}",
        f"Edges: {features.get('edge_ratio', 0.0):.2f}",
        f"Potholes: {len(potholes.boxes)}",
    ]

    x = max(
        frame.shape[1] - 150,
        8,
    )

    y = 16

    for line in debug_lines:
        # Dark outline.
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        y += 17


def annotate_frame(
    frame: np.ndarray,
    surface: SurfaceResult,
    potholes: PotholeResult,
    action: VehicleAction,
    road_y_offset: int,
    debug: bool = False,
) -> np.ndarray:
    """
    Create the final compact driver-assistance display.
    """
    output = frame.copy()

    # Visual monitoring boundary only.
    draw_visual_monitoring_region(
        output
    )

    # Actual pothole coordinates remain tied to the 0.55 crop.
    draw_potholes(
        output,
        potholes,
        road_y_offset,
    )

    draw_status_panel(
        output,
        surface,
        potholes,
        action,
    )

    if debug:
        draw_debug_information(
            output,
            surface,
            potholes,
        )

    return output