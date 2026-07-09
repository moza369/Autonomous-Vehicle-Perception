import cv2
import numpy as np
# Importation depuis notre package modulaire
from detector import ConstructionZoneDetector

def create_mock_scene():
    """Crée une route fictive simulée pour tester l'API."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Fond de route gris
    cv2.rectangle(img, (0, 200), (640, 480), (100, 100, 100), -1)
    
    # Marquage classique blanc
    cv2.line(img, (150, 480), (280, 200), (255, 255, 255), 3) # Gauche
    cv2.line(img, (490, 480), (360, 200), (255, 255, 255), 3) # Droite

    # Alignement de cônes orange (Merge Left)
    cone_positions = [
        (480, 430, 25, 40),
        (430, 370, 20, 32),
        (380, 310, 16, 26),
        (330, 250, 12, 20),
    ]
    for (x, y, w, h) in cone_positions:
        # Dessiner le corps du cône
        pts = np.array([[x, y + h], [x + w, y + h], [x + w // 2, y]], np.int32)
        cv2.fillPoly(img, [pts], (0, 110, 240)) # BGR Orange
        # Base du cône
        cv2.rectangle(img, (x - 2, y + h - 5), (x + w + 2, y + h), (30, 30, 30), -1)
        # Bande blanche
        cv2.rectangle(img, (x + w//4, y + h//3), (x + 3*w//4, y + 2*h//3), (255, 255, 255), -1)
        
    return img

if __name__ == "__main__":
    frame = create_mock_scene()
    
    # Instanciation de notre module d'équipe
    detector = ConstructionZoneDetector()
    
    # Exécution du traitement
    annotated_frame, merge_required, metadata = detector.process_image(frame)
    
    # Affichage des métadonnées de sortie pour l'intégration globale
    print("-- RÉSULTATS DU MODULE CONSTRUCTION ZONE --")
    print(f"Cônes détectés : {metadata['detected_cones_count']}")
    print(f"Coordonnées au sol : {metadata['cones_coordinates']}")
    print(f"Pente calculée : {metadata['slope']:.4f}")
    print(f"Décision de rabattement requise (Merge Required) : {merge_required}")
    
    if merge_required:
        print("ACTION ADAS : Désactiver le Lane Assist standard et bifurquer à gauche.")
    else:
        print("ACTION ADAS : Maintien de voie standard actif.")
        
    # Enregistrer le résultat pour vérification visuelle
    cv2.imwrite("integrated_demo_result.png", annotated_frame)
    print("Image de démo sauvegardée sous 'integrated_demo_result.png'.")
