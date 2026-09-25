import cv2
import mediapipe as mp
import sys
import threading
import time


class SafetyAlarm:
    """Non-blocking warning chime with a one-second repeat interval."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._active = False

    def play(self):
        if not self.enabled or self._active:
            return
        self._active = True
        threading.Thread(target=self._beep, daemon=True).start()

    def _beep(self):
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(1000, 300)
            else:
                print("\a", end="", flush=True)
        finally:
            time.sleep(1.0)
            self._active = False


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return {"leaning": False, "feet_on_dashboard": False, "out_of_position": False, "label": "No pose detected"}
        self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        nose = lm[0]
        l_shoulder = lm[11]
        r_shoulder = lm[12]
        l_hip = lm[23]
        r_hip = lm[24]
        l_ankle = lm[27]
        r_ankle = lm[28]
        shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2
        shoulder_center_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_center_y = (l_hip.y + r_hip.y) / 2
        forward = nose.y > shoulder_center_y + 0.05
        sideways = abs(nose.x - shoulder_center_x) > 0.12
        leaning = forward or sideways
        feet_on_dashboard = False
        for ankle in [l_ankle, r_ankle]:
            if ankle.visibility > 0.3 and ankle.y < hip_center_y:
                feet_on_dashboard = True
        out_of_position = leaning or feet_on_dashboard
        label = "SAFE POSTURE"
        if out_of_position:
            parts = []
            if forward: parts.append("LEANING FORWARD")
            if sideways: parts.append("LEANING SIDEWAYS")
            if feet_on_dashboard: parts.append("FEET ON DASHBOARD")
            label = " | ".join(parts)
        return {"leaning": leaning, "feet_on_dashboard": feet_on_dashboard, "out_of_position": out_of_position, "label": label}

class SeatbeltDetector:
    def __init__(self):
        self.status = "ON"
    def toggle(self):
        self.status = "OFF" if self.status == "ON" else "ON"
    def get_status(self):
        return self.status

class SafetySystem:
    def __init__(self, current_speed=80):
        self.current_speed = current_speed
        self.state = "NORMAL"
        self.airbag = "ENABLED"
        self.speed_limit = None
    def update(self, seatbelt, posture_info):
        belt_off = seatbelt == "OFF"
        bad_posture = posture_info["out_of_position"]
        if belt_off and bad_posture:
            self.state = "DANGER"
            self.airbag = "DISABLED"
            self.speed_limit = self.current_speed
        elif belt_off:
            self.state = "WARNING"
            self.airbag = "ENABLED"
            self.speed_limit = self.current_speed
        elif bad_posture:
            self.state = "WARNING"
            self.airbag = "ENABLED"
            self.speed_limit = None
        else:
            self.state = "NORMAL"
            self.airbag = "ENABLED"
            self.speed_limit = None
        return self.state
    def get_airbag_status(self): return self.airbag
    def get_speed_limit(self): return self.speed_limit

class PassengerSafetyDetector:
    def __init__(self, speed=80, audio_enabled=True):
        self.pose_detector = PoseDetector()
        self.seatbelt_detector = SeatbeltDetector()
        self.safety_system = SafetySystem(current_speed=speed)
        self.alarm = SafetyAlarm(enabled=audio_enabled)
        
    def reset(self):
        self.seatbelt_detector.status = "ON"
        self.safety_system.state = "NORMAL"
        self.safety_system.airbag = "ENABLED"
        self.safety_system.speed_limit = None
        
    def toggle_seatbelt(self):
        self.seatbelt_detector.toggle()
        
    def process_frame(self, frame):
        annotated_frame = frame.copy()
        posture_info = self.pose_detector.detect(annotated_frame)
        seatbelt_status = self.seatbelt_detector.get_status()
        state = self.safety_system.update(seatbelt_status, posture_info)

        belt_off = seatbelt_status == "OFF"
        bad_posture = posture_info["out_of_position"]
        warning = "NORMAL"
        action = "NORMAL PASSENGER SAFETY OPERATION"

        if belt_off and bad_posture:
            warning = "WARNING: PASSENGER UNBUCKLED & OUT OF POSITION"
            action = (
                "ACTION: PASSENGER AIRBAG DISABLED - "
                f"SPEED RESTRICTED TO {self.safety_system.current_speed} KM/H"
            )
        elif belt_off:
            warning = "WARNING: PASSENGER UNBUCKLED"
        elif bad_posture:
            warning = "WARNING: PASSENGER OUT OF POSITION"

        if state != "NORMAL":
            self.alarm.play()
        
        telemetry = {
            "posture": posture_info["label"],
            "seatbelt": seatbelt_status,
            "system_state": state,
            "airbag": self.safety_system.get_airbag_status(),
            "speed_limit": self.safety_system.get_speed_limit(),
            "warning": warning,
            "action": action,
        }
        
        color = (0, 255, 0)
        if state == "WARNING":
            color = (0, 165, 255)
        elif state == "DANGER":
            color = (0, 0, 255)
            
        cv2.putText(annotated_frame, f"State: {state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated_frame, f"Airbag: {self.safety_system.get_airbag_status()}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        speed_lim = self.safety_system.get_speed_limit()
        speed_str = f"{speed_lim} km/h" if speed_lim else "None"
        cv2.putText(annotated_frame, f"Speed Limit: {speed_str}", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Seatbelt: {seatbelt_status}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Posture: {posture_info['label']}", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        h, w = annotated_frame.shape[:2]
        if state != "NORMAL":
            banner_color = (0, 0, 180) if state == "DANGER" else (0, 140, 220)
            cv2.rectangle(annotated_frame, (0, h - 115), (w, h), banner_color, -1)
            cv2.putText(
                annotated_frame,
                warning,
                (15, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if state == "DANGER":
                cv2.putText(
                    annotated_frame,
                    action,
                    (15, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return annotated_frame, telemetry
