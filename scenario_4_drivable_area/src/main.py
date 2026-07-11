import cv2
import numpy as np
import onnxruntime as ort
import os
import random


#  Définition des chemins relatifs




BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # src/
PROJECT_DIR = os.path.dirname(BASE_DIR)                    # scenario_4_drivable_area/


DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "bdd100k", "images", "100k", "train")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "yolov8n-seg.onnx")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


os.makedirs(RESULTS_DIR, exist_ok=True)




#  Sélection aléatoire d'une image du dataset




image_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]
if len(image_files) == 0:
    raise FileNotFoundError("Aucune image .jpg trouvée dans le dataset.")


random_image = random.choice(image_files)
image_path = os.path.join(DATASET_DIR, random_image)


print(f"Image sélectionnée : {random_image}")


img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Impossible de lire l'image sélectionnée")


h, w = img.shape[:2]


# Charger le modèle YOLO ONNX




session = ort.InferenceSession(MODEL_PATH)


# Préparation de l'image pour YOLO


img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.transpose(2, 0, 1)[None].astype(np.float32)




# Inférence YOLO


outputs = session.run(None, {"images": img_input})
masks = outputs[1][0]
mask = masks[0]


mask_resized = cv2.resize(mask, (w, h))
mask_binary = (mask_resized > 0.5).astype(np.uint8)


mask_vis = (mask_binary * 255).astype(np.uint8)
colored_mask = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)


overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)




# Analyse de dérive (steering)


ys, xs = np.where(mask_binary == 1)
action_text = "ACTION: KEEP LANE - CENTERED"


if len(xs) > 0:
    lane_center_x = int(np.mean(xs))
    image_center_x = w // 2
    tolerance = w * 0.05


    if lane_center_x > image_center_x + tolerance:
        action_text = "ACTION: STEERING LEFT - CORRECTING LANE DEPARTURE"
    elif lane_center_x < image_center_x - tolerance:
        action_text = "ACTION: STEERING RIGHT - CORRECTING LANE DEPARTURE"


    cv2.line(overlay, (image_center_x, 0), (image_center_x, h), (0, 255, 0), 2)
    cv2.line(overlay, (lane_center_x, 0), (lane_center_x, h), (0, 0, 255), 2)


# Affichage et sauvegarde




cv2.putText(
    overlay,
    action_text,
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)


output_path = os.path.join(RESULTS_DIR, f"result_{random_image}")
cv2.imwrite(output_path, overlay)


print(f"Résultat sauvegardé dans : {output_path}")


cv2.imshow("Drivable Area & Action", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

