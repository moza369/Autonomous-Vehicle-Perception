import cv2
import numpy as np


class ConstructionZoneDetector:
    """
    Détecteur simple de zone de chantier.
    Il détecte les cônes orange et affiche l'action demandée.
    """

    def __init__(self, min_cone_area=300):
        # Surface minimale pour accepter un cône
        self.min_cone_area = min_cone_area

        # Couleur orange en HSV
        self.orange_lower1 = np.array([0, 70, 70])
        self.orange_upper1 = np.array([25, 255, 255])

        self.orange_lower2 = np.array([160, 70, 70])
        self.orange_upper2 = np.array([180, 255, 255])

    def detect_cones(self, frame):
        """
        Détecter les cônes orange dans l'image.
        """
        h_img, w_img = frame.shape[:2]

        # Réduire le bruit
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convertir l'image en HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Créer le masque orange
        mask1 = cv2.inRange(hsv, self.orange_lower1, self.orange_upper1)
        mask2 = cv2.inRange(hsv, self.orange_lower2, self.orange_upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Ignorer le haut de l'image
        mask[:int(h_img * 0.18), :] = 0

        # Nettoyer le masque
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cones = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignorer les petits objets
            if area < self.min_cone_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ignorer les objets trop petits
            if w < 7 or h < 10:
                continue

            # Ignorer les objets trop grands
            if w > 180 or h > 240:
                continue

            # Vérifier la forme verticale
            aspect_ratio = h / float(w)

            if aspect_ratio < 0.5 or aspect_ratio > 5.8:
                continue

            # Calculer le centre du bas du cône
            cx = int(x + w / 2)
            cy = int(y + h)

            cones.append({
                "box": (x, y, w, h),
                "ground_center": (cx, cy),
                "area": area
            })

        # Trier par taille et garder seulement les cônes principaux
        cones = sorted(cones, key=lambda c: c["area"], reverse=True)
        cones = cones[:7]

        # Trier du plus loin au plus proche
        cones = sorted(cones, key=lambda c: c["ground_center"][1])

        return cones

    def analyze_scene(self, cones):
        """
        Si plusieurs cônes sont détectés, il y a une zone de chantier.
        """
        return len(cones) >= 3

    def draw_path_line(self, output_frame, cones):
        """
        Dessiner une ligne qui suit les cônes détectés.
        """
        if len(cones) < 2:
            return None, 0.0

        points = np.array([cone["ground_center"] for cone in cones], dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        vx = float(vx[0])
        vy = float(vy[0])
        x0 = float(x0[0])
        y0 = float(y0[0])

        slope = vx / vy if abs(vy) > 1e-6 else 0.0

        h_img, w_img = output_frame.shape[:2]

        y1 = int(np.min(points[:, 1]))
        y2 = int(np.max(points[:, 1]))

        x1 = int(x0 + slope * (y1 - y0))
        x2 = int(x0 + slope * (y2 - y0))

        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img - 1, x2))

        fitted_line = ((x1, y1), (x2, y2))

        cv2.line(output_frame, fitted_line[0], fitted_line[1], (0, 255, 255), 3)

        return fitted_line, slope

    def draw_overlay(self, output_frame, merge_required):
        """
        Afficher le message demandé par la question 6.
        """
        h, w = output_frame.shape[:2]

        cv2.rectangle(output_frame, (0, 0), (w, 88), (0, 0, 0), -1)

        if merge_required:
            cv2.putText(
                output_frame,
                "ACTION: CONSTRUCTION ZONE - MERGING LEFT",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

            cv2.putText(
                output_frame,
                "LANE KEEPING DISABLED - FOLLOWING CONES",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                2
            )
        else:
            cv2.putText(
                output_frame,
                "NO CONSTRUCTION MERGE DETECTED",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    def process_image(self, frame):
        """
        Traiter l'image et retourner le résultat.
        """
        if frame is None:
            raise ValueError("Image vide.")

        output_frame = frame.copy()

        # Détecter les cônes
        cones = self.detect_cones(frame)

        # Prendre la décision
        merge_required = self.analyze_scene(cones)

        # Dessiner les cônes
        for cone in cones:
            x, y, w, h = cone["box"]
            cx, cy = cone["ground_center"]

            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(output_frame, (cx, cy), 5, (0, 255, 0), -1)

            label_y = max(105, y - 5)

            cv2.putText(
                output_frame,
                "CONE",
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1
            )

        # Dessiner la ligne
        fitted_line, slope = self.draw_path_line(output_frame, cones)

        # Afficher l'action
        self.draw_overlay(output_frame, merge_required)

        metadata = {
            "detected_cones_count": len(cones),
            "cones_coordinates": [c["ground_center"] for c in cones],
            "fitted_line": fitted_line,
            "slope": slope,
            "construction_zone_detected": merge_required,
            "lane_keeping_disabled": merge_required,
            "system_action": (
                "ACTION: CONSTRUCTION ZONE - MERGING LEFT"
                if merge_required
                else "NO CONSTRUCTION MERGE DETECTED"
            )
        }

        return output_frame, merge_required, metadata