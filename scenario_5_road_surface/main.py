"""
Team 5 - Road Surface Condition Detection

Detects:
- Wet roads / standing water
- Snow-covered roads
- Icy-looking roads
- Large potholes

Supports:
- Images
- Videos
- Webcam

This is a classroom prototype and must not control a real vehicle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pothole_detector import detect_large_potholes
from road_surface_detector import detect_road_surface
from utils import (
    determine_vehicle_action,
    get_road_region,
)
from visualization import annotate_frame


def analyze_frame(
    frame,
    debug: bool = False,
):
    """Run the full perception and decision pipeline."""
    road_region, y_offset = get_road_region(
        frame
    )

    surface, processed = detect_road_surface(
        road_region
    )

    potholes = detect_large_potholes(
        road_region,
        processed,
    )

    action = determine_vehicle_action(
        surface,
        potholes,
    )

    annotated = annotate_frame(
        frame=frame,
        surface=surface,
        potholes=potholes,
        action=action,
        road_y_offset=y_offset,
        debug=debug,
    )

    return (
        annotated,
        surface,
        potholes,
        action,
        processed,
    )


def process_image(
    input_path: str,
    output_path: str,
    debug: bool,
) -> None:
    """Process one road image."""
    frame = cv2.imread(input_path)

    if frame is None:
        raise FileNotFoundError(
            f"Could not open image: {input_path}"
        )

    (
        annotated,
        surface,
        potholes,
        action,
        processed,
    ) = analyze_frame(
        frame,
        debug,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_file),
        annotated,
    ):
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )

    print()
    print("TEAM 5 ROAD SURFACE RESULT")
    print("------------------------------------------")
    print(f"Surface: {surface.condition}")
    print(
        f"Confidence: "
        f"{surface.confidence * 100:.1f}%"
    )
    print(
        f"Large potholes: "
        f"{len(potholes.boxes)}"
    )
    print(f"Warning: {action.warning}")
    print(f"Action: {action.action}")
    print(
        f"Traction control: "
        f"{action.traction_control}"
    )
    print(
        f"Maximum speed: "
        f"{action.maximum_speed} KM/H"
    )
    print(f"Saved result: {output_path}")
    print("------------------------------------------")

    cv2.imshow(
        "Road Surface Detection",
        annotated,
    )

    if debug:
        cv2.imshow(
            "Canny Edges",
            processed["edges"],
        )

        cv2.imshow(
            "Morphological Closing",
            processed["closed_edges"],
        )

        cv2.imshow(
            "Large Pothole Mask",
            potholes.mask,
        )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(
    input_source: str | int,
    output_path: str | None,
    debug: bool,
) -> None:
    """Process a video or webcam feed."""
    capture = cv2.VideoCapture(
        input_source
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open source: {input_source}"
        )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    writer = None

    if output_path is not None:
        output_file = Path(output_path)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = cv2.VideoWriter(
            str(output_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()

            raise RuntimeError(
                f"Could not create video: {output_path}"
            )

    print("Video processing started.")
    print("Press Q inside the video window to stop.")

    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            (
                annotated,
                surface,
                potholes,
                action,
                processed,
            ) = analyze_frame(
                frame,
                debug,
            )

            if writer is not None:
                writer.write(annotated)

            cv2.imshow(
                "Road Surface Detection",
                annotated,
            )

            if debug:
                cv2.imshow(
                    "Canny Edges",
                    processed["edges"],
                )

                cv2.imshow(
                    "Large Pothole Mask",
                    potholes.mask,
                )

            frame_number += 1

            if frame_number % 30 == 0:
                print(
                    f"Frame {frame_number}: "
                    f"surface={surface.condition}, "
                    f"potholes={len(potholes.boxes)}, "
                    f"max_speed={action.maximum_speed}"
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    print("Video processing finished.")

    if output_path is not None:
        print(f"Saved video: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Team 5 road surface condition detection."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "image",
            "video",
            "camera",
        ],
        default="image",
    )

    parser.add_argument(
        "--input",
        default="input/wet_road.jpg",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    try:
        if args.mode == "image":
            output_path = (
                args.output
                or "output/road_surface_result.jpg"
            )

            process_image(
                input_path=args.input,
                output_path=output_path,
                debug=args.debug,
            )

        elif args.mode == "video":
            output_path = (
                args.output
                or "output/road_surface_result.mp4"
            )

            process_video(
                input_source=args.input,
                output_path=output_path,
                debug=args.debug,
            )

        else:
            process_video(
                input_source=0,
                output_path=None,
                debug=args.debug,
            )

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()