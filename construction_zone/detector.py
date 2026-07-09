import cv2
import numpy as np

class ConstructionZoneDetector:
    """
    Module de détection de zone de chantier pour le système ADAS.
    Détecte les cônes orange et prend la décision d'évitement / rabattement à gauche.
    """
    def __init__(self, min_cone_area=30, aspect_ratio_range=(1.0, 3.0), solidity_threshold=0.6):
        # Paramètres de filtrage réglables
        self.min_cone_area = min_cone_area
        self.aspect_ratio_range = aspect_ratio_range
        self.solidity_threshold = solidity_threshold

        # Seuils HSV pour la détection de la couleur orange
        self.orange_lower1 = np.array([0, 120, 100])
        self.orange_upper1 = np.array([15, 255, 255])
        
        self.orange_lower2 = np.array([165, 120, 100])
        self.orange_upper2 = np.array([180, 255, 255])

    def detect_cones(self, hsv_frame):
        """
        Segment et filtre les cônes de chantier orange dans l'image.
        """
        # Création des masques binaires de couleur
        mask1 = cv2.inRange(hsv_frame, self.orange_lower1, self.orange_upper1)
        mask2 = cv2.inRange(hsv_frame, self.orange_lower2, self.orange_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Opérations morphologiques : Fermeture verticale pour relier le haut/bas coupé par la bande blanche
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Détection des contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cones_metadata = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_cone_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(h) / w
            
            # Filtrage géométrique
            if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                if solidity > self.solidity_threshold:
                    cx = int(x + w / 2)
                    cy = int(y + h) # Point d'ancrage au sol (bas du cône)
                    
                    cones_metadata.append({
                        'box': (x, y, w, h),
                        'ground_center': (cx, cy),
                        'contour': contour,
                        'area': area
                    })
                    
        return cones_metadata, mask

    def analyze_trajectory(self, cones, img_shape):
        """
        Ajuste une ligne sur la base des cônes et détermine si un rabattement est requis.
        """
        h_img, w_img = img_shape[:2]
        merge_required = False
        fitted_line_points = None
        slope = 0.0

        if len(cones) >= 2:
            pts = np.array([cone['ground_center'] for cone in cones])
            # Trier du haut vers le bas de l'image (Y croissant)
            pts = pts[np.argsort(pts[:, 1])]
            
            # Ajustement linéaire (Régression L2)
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vx[0] / vy[0] if vy[0] != 0 else 0.0
            
            # Délimitation de la droite sur la zone d'intérêt (bas de l'image à la ligne d'horizon simulée)
            y1 = int(h_img * 0.4)
            y2 = int(h_img)
            x1 = int(x0[0] + slope * (y1 - y0[0]))
            x2 = int(x0[0] + slope * (y2 - y0[0]))
            
            fitted_line_points = ((x1, y1), (x2, y2))
            
            # ANALYSE DE LA TRAJECTOIRE :
            # Dans le repère écran (Y vers le bas), une ligne de cônes partant du bas-droite (proche)
            # et montant vers le haut-gauche (loin) possède une pente (dx/dy) positive.
            # Si cette pente dépasse 0.1, les cônes de droite obstruent notre voie -> Rabattement à gauche requis.
            if slope > 0.1:
                merge_required = True

        return merge_required, fitted_line_points, slope

    def process_image(self, frame):
        """
        Méthode principale d'API : traite une frame d'entrée et renvoie l'image annotée,
        l'état de décision (merge_required) et les métadonnées pour l'intégration.
        """
        output_frame = frame.copy()
        
        # Prétraitement
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Détection
        cones, mask = self.detect_cones(hsv)
        
        # Analyse de trajectoire
        merge_required, fitted_line, slope = self.analyze_trajectory(cones, frame.shape)
        
        # Annotations de l'IHM
        for cone in cones:
            x, y, w, h = cone['box']
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.circle(output_frame, cone['ground_center'], 5, (0, 255, 0), -1)
            cv2.putText(output_frame, "CONE", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
        if fitted_line:
            cv2.line(output_frame, fitted_line[0], fitted_line[1], (0, 165, 255), 3)

        metadata = {
            'detected_cones_count': len(cones),
            'cones_coordinates': [c['ground_center'] for c in cones],
            'fitted_line': fitted_line,
            'slope': slope
        }
        
        return output_frame, merge_required, metadata
