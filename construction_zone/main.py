import cv2
import os
from detector import ConstructionZoneDetector


def main():
    print("Using detector version: clean construction zone detector")

    # Chercher l'image de test
    image_path = os.path.join("test_images", "construction_zone_real.jpg")

    if not os.path.exists(image_path):
        image_path = os.path.join("test_images", "construction_zone_real.jpeg")

    if not os.path.exists(image_path):
        print("Erreur : image introuvable.")
        print("Mets l'image dans test_images sous le nom construction_zone_real.jpg ou construction_zone_real.jpeg")
        return

    # Créer le dossier des résultats
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Chemin de l'image de sortie
    output_path = os.path.join(results_dir, "output_construction_zone_real.png")

    # Lire l'image
    frame = cv2.imread(image_path)

    if frame is None:
        print("Erreur : impossible de lire l'image.")
        print("Chemin :", image_path)
        return

    # Redimensionner l'image
    frame = cv2.resize(frame, (640, 480))

    # Créer le détecteur
    detector = ConstructionZoneDetector(min_cone_area=390)

    # Traiter l'image
    annotated_frame, merge_required, metadata = detector.process_image(frame)

    # Afficher les résultats dans le terminal
    print("-- TEST IMAGE RÉELLE : ZONE DE CHANTIER --")
    print("Image utilisée :", image_path)
    print("Cônes détectés :", metadata["detected_cones_count"])
    print("Coordonnées des cônes :", metadata["cones_coordinates"])
    print("Merge Required :", merge_required)
    print(metadata["system_action"])

    if merge_required:
        print("SYSTEM: STANDARD LANE KEEPING DISABLED - FOLLOWING CONES")

    # Sauvegarder l'image annotée
    cv2.imwrite(output_path, annotated_frame)
    print("Image annotée sauvegardée sous :", output_path)

    # Afficher l'image annotée
    cv2.imshow("Construction Zone Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()