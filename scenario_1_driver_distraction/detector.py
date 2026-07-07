import cv2
import numpy as np
import mediapipe as mp

from config import DOWN_SCORE_THRESHOLD, EAR_THRESHOLD


# ============================================================
# Initialisation de MediaPipe Face Mesh
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# Fonctions utilitaires
# ============================================================

def point_to_pixel(landmark, width, height):
    """
    Convertit un point MediaPipe normalisé en coordonnées pixels.

    MediaPipe donne les coordonnées entre 0 et 1.
    OpenCV travaille avec des coordonnées en pixels.
    """
    return np.array([
        int(landmark.x * width),
        int(landmark.y * height)
    ])


def calculate_eye_aspect_ratio(face_landmarks, eye_indices, width, height):
    """
    Calcule le Eye Aspect Ratio pour un œil.

    EAR = distance verticale moyenne / distance horizontale

    Si EAR est faible, l'œil est presque fermé ou le regard n'est pas
    correctement dirigé vers la route.
    """

    p1 = point_to_pixel(face_landmarks.landmark[eye_indices["left_corner"]], width, height)
    p2 = point_to_pixel(face_landmarks.landmark[eye_indices["right_corner"]], width, height)

    p3 = point_to_pixel(face_landmarks.landmark[eye_indices["top_1"]], width, height)
    p4 = point_to_pixel(face_landmarks.landmark[eye_indices["bottom_1"]], width, height)

    p5 = point_to_pixel(face_landmarks.landmark[eye_indices["top_2"]], width, height)
    p6 = point_to_pixel(face_landmarks.landmark[eye_indices["bottom_2"]], width, height)

    horizontal_distance = np.linalg.norm(p1 - p2)
    vertical_distance_1 = np.linalg.norm(p3 - p4)
    vertical_distance_2 = np.linalg.norm(p5 - p6)

    if horizontal_distance == 0:
        return 0.0

    ear = (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)

    return ear


# ============================================================
# Fonction principale de détection
# ============================================================

def detect_driver_attention(frame):
    """
    Détecte si le conducteur est attentif ou distrait.

    La fonction combine deux critères :

    1. down_score :
       mesure si la tête est inclinée vers le bas.

    2. average_ear :
       mesure l'ouverture des yeux.

    Retour :
        status : état détecté
        down_score : score de tête baissée
        average_ear : score d'ouverture des yeux
        annotated_frame : image annotée
    """

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    annotated_frame = frame.copy()

    if not results.multi_face_landmarks:
        return "NO FACE DETECTED", None, None, annotated_frame

    height, width, _ = frame.shape
    face_landmarks = results.multi_face_landmarks[0]

    # --------------------------------------------------------
    # 1. Calcul du down_score
    # --------------------------------------------------------

    front = face_landmarks.landmark[10]
    nose = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    front_x, front_y = int(front.x * width), int(front.y * height)
    nose_x, nose_y = int(nose.x * width), int(nose.y * height)
    chin_x, chin_y = int(chin.x * width), int(chin.y * height)
    left_eye_x, left_eye_y = int(left_eye.x * width), int(left_eye.y * height)
    right_eye_x, right_eye_y = int(right_eye.x * width), int(right_eye.y * height)

    eye_center_y = int((left_eye_y + right_eye_y) / 2)
    face_height = chin_y - front_y

    if face_height <= 0:
        return "INVALID FACE GEOMETRY", None, None, annotated_frame

    nose_eye_distance = nose_y - eye_center_y
    down_score = nose_eye_distance / face_height

    # --------------------------------------------------------
    # 2. Calcul du Eye Aspect Ratio
    # --------------------------------------------------------

    left_eye_indices = {
        "left_corner": 33,
        "right_corner": 133,
        "top_1": 159,
        "bottom_1": 145,
        "top_2": 158,
        "bottom_2": 153
    }

    right_eye_indices = {
        "left_corner": 362,
        "right_corner": 263,
        "top_1": 386,
        "bottom_1": 374,
        "top_2": 385,
        "bottom_2": 380
    }

    left_ear = calculate_eye_aspect_ratio(
        face_landmarks,
        left_eye_indices,
        width,
        height
    )

    right_ear = calculate_eye_aspect_ratio(
        face_landmarks,
        right_eye_indices,
        width,
        height
    )

    average_ear = (left_ear + right_ear) / 2.0

    # --------------------------------------------------------
    # 3. Décision finale
    # --------------------------------------------------------

    head_down = down_score > DOWN_SCORE_THRESHOLD
    eyes_not_on_road = average_ear < EAR_THRESHOLD

    if head_down or eyes_not_on_road:
        status = "WARNING: EYES ON ROAD"
        color = (0, 0, 255)
    else:
        status = "DRIVER ATTENTIVE"
        color = (0, 255, 0)

    # --------------------------------------------------------
    # 4. Annotation de l'image
    # --------------------------------------------------------

    important_points = [
        (front_x, front_y, "front"),
        (nose_x, nose_y, "nose"),
        (chin_x, chin_y, "chin"),
        (left_eye_x, left_eye_y, "left_eye"),
        (right_eye_x, right_eye_y, "right_eye")
    ]

    for x, y, label in important_points:
        cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(
            annotated_frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

    cv2.putText(
        annotated_frame,
        status,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        annotated_frame,
        f"Down score: {down_score:.3f}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"EAR: {average_ear:.3f}",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    return status, down_score, average_ear, annotated_frame
