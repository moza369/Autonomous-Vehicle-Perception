import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        info = {
            "leaning": False,
            "feet_on_dashboard": False,
            "out_of_position": False,
            "label": "SAFE"
        }

        if results.pose_landmarks:
            # Dessiner le squelette
            self.drawer.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Calcul des centres pour reference
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2

            # Detection du buste penche en avant ou sur les cotes
            leaning = False
            if nose.y > shoulder_center_y + 0.05:
                leaning = True
            if abs(nose.x - shoulder_center_x) > 0.12:
                leaning = True

            # Detection des pieds sur le tableau de bord (chevilles plus hautes que les hanches)
            feet_up = False
            if left_ankle.visibility > 0.3 and left_ankle.y < hip_center_y:
                feet_up = True
            if right_ankle.visibility > 0.3 and right_ankle.y < hip_center_y:
                feet_up = True

            info["leaning"] = leaning
            info["feet_on_dashboard"] = feet_up
            info["out_of_position"] = leaning or feet_up

            # Label d'affichage
            if leaning and feet_up:
                info["label"] = "OUT OF POSITION (LEANING + FEET ON DASHBOARD)"
            elif leaning:
                info["label"] = "OUT OF POSITION (LEANING FORWARD)"
            elif feet_up:
                info["label"] = "OUT OF POSITION (FEET ON DASHBOARD)"
            else:
                info["label"] = "SAFE"

        return info
