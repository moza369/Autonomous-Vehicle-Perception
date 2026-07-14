import cv2
import os
from detector import ConstructionZoneDetector


def create_detector():
    """
    Créer le détecteur.
    On garde une valeur moyenne pour éviter trop de fausses détections.
    """
    return ConstructionZoneDetector(min_cone_area=320)


def process_image():
    """
    Tester le module avec une image réelle.
    """
    image_path = os.path.join("test_images", "construction_zone_real.jpg")

    if not os.path.exists(image_path):
        image_path = os.path.join("test_images", "construction_zone_real.jpeg")

    if not os.path.exists(image_path):
        print("Erreur : image introuvable.")
        print("Mets l'image dans test_images/construction_zone_real.jpg")
        return

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "output_image_result.png")

    frame = cv2.imread(image_path)

    if frame is None:
        print("Erreur : impossible de lire l'image.")
        return

    frame = cv2.resize(frame, (640, 480))

    detector = create_detector()

    annotated_frame, merge_required, metadata = detector.process_image(frame)

    print("\n--- TEST IMAGE ---")
    print("Image utilisée :", image_path)
    print("Cônes détectés :", metadata["detected_cones_count"])
    print("Merge Required :", merge_required)
    print(metadata["system_action"])

    cv2.imwrite(output_path, annotated_frame)
    print("Image résultat sauvegardée :", output_path)

    cv2.imshow("Image Test - Construction Zone", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video():
    """
    Tester le module avec une vidéo.
    Le programme lit la vidéo frame par frame comme un stream.
    """
    video_path = os.path.join("test_videos", "construction_zone_video.mp4")

    if not os.path.exists(video_path):
        print("Erreur : vidéo introuvable.")
        print("Mets la vidéo dans test_videos/construction_zone_video.mp4")
        return

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "output_video_result.mp4")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    detector = create_detector()

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    width = 640
    height = 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    print("\n--- TEST VIDÉO ---")
    print("Vidéo utilisée :", video_path)
    print("Appuie sur q pour arrêter la vidéo.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        annotated_frame, merge_required, metadata = detector.process_image(frame)

        frame_count += 1

        # Afficher une petite information dans le terminal chaque 20 frames
        if frame_count % 20 == 0:
            print(
                f"Frame {frame_count} | "
                f"Cônes : {metadata['detected_cones_count']} | "
                f"Merge Required : {merge_required}"
            )

        writer.write(annotated_frame)

        cv2.imshow("Video Stream - Construction Zone", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("Vidéo résultat sauvegardée :", output_path)


def process_camera():
    """
    Tester le module avec la webcam.
    C'est optionnel.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : caméra non disponible.")
        return

    detector = create_detector()

    print("\n--- TEST CAMÉRA ---")
    print("Appuie sur q pour arrêter.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        annotated_frame, merge_required, metadata = detector.process_image(frame)

        cv2.imshow("Camera Stream - Construction Zone", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Menu simple pour choisir le test.
    """
    print("Construction Zone Detection")
    print("1 - Tester avec image")
    print("2 - Tester avec vidéo")
    print("3 - Tester avec caméra")
    print("4 - Tester image puis vidéo")

    choice = input("Choisis une option : ")

    if choice == "1":
        process_image()
    elif choice == "2":
        process_video()
    elif choice == "3":
        process_camera()
    elif choice == "4":
        process_image()
        process_video()
    else:
        print("Choix invalide.")


if __name__ == "__main__":
    main()