import cv2
import time

from config import (
    CAMERA_INDEX,
    WARNING_DURATION_LIMIT,
    INITIAL_SPEED,
    MIN_SPEED,
    SPEED_DECREASE_STEP
)

from detector import detect_driver_attention


def run_driver_distraction_system():
    """
    Lance le système de détection de distraction du conducteur.

    Scénario traité :
    - Le véhicule roule à 60 km/h.
    - Si le conducteur regarde vers le bas, le système affiche :
      WARNING: EYES ON ROAD.
    - Si la distraction dure plus de 3 secondes, le système affiche :
      ACTION: DISENGAGING CRUISE CONTROL
      SLOWING DOWN.
    - La vitesse simulée diminue progressivement jusqu'à 30 km/h.
    """

    distraction_start_time = None
    limp_mode_active = False
    vehicle_speed = INITIAL_SPEED

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    print("Webcam ouverte.")
    print("Regarde vers le bas pendant plus de 3 secondes pour tester l'escalade.")
    print("Appuie sur la touche 'q' pour quitter.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur : impossible de lire la frame.")
            break

        status, down_score, average_ear, annotated_frame = detect_driver_attention(frame)

        current_time = time.time()

        if status == "WARNING: EYES ON ROAD":

            if distraction_start_time is None:
                distraction_start_time = current_time

            distraction_duration = current_time - distraction_start_time

            cv2.putText(
                annotated_frame,
                f"Distraction time: {distraction_duration:.1f}s",
                (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            if distraction_duration >= WARNING_DURATION_LIMIT:
                limp_mode_active = True

        else:
            distraction_start_time = None
            limp_mode_active = False
            vehicle_speed = INITIAL_SPEED

        if limp_mode_active:
            vehicle_speed = max(
                MIN_SPEED,
                vehicle_speed - SPEED_DECREASE_STEP
            )

            cv2.putText(
                annotated_frame,
                "ACTION: DISENGAGING CRUISE CONTROL",
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                "SLOWING DOWN",
                (30, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            annotated_frame,
            f"Speed: {vehicle_speed:.1f} km/h",
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.imshow(
            "Driver Distraction Detection - Scenario 1",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Test terminé.")


if __name__ == "__main__":
    run_driver_distraction_system()