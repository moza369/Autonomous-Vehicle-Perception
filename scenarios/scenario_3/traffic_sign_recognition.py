import cv2
from ultralytics import YOLO


class VehicleState:
    def __init__(self):
        self.speed_kmh = 50.0
        self.friction = 0.7
        self.g = 9.81
        self.reaction_time = 1.5

    def get_stopping_distance(self):
        speed_ms = self.speed_kmh / 3.6
        reaction_dist = speed_ms * self.reaction_time
        braking_dist = (speed_ms ** 2) / (2 * self.friction * self.g)
        return reaction_dist + braking_dist

class DecisionEngine:
    def __init__(self):
        self.action = "ACTION: CONTINUE DRIVING"
        
    def determine_action(self, detected_classes):
        if "traffic_light_red" in detected_classes:
            self.action = "ACTION: RED LIGHT DETECTED - APPLYING BRAKES"
        elif "stop" in detected_classes:
            self.action = "ACTION: STOP SIGN DETECTED - APPLYING BRAKES"
        elif "traffic_light_yellow" in detected_classes:
            self.action = "WARNING: YELLOW LIGHT DETECTED - PREPARE TO STOP"
        elif "speed_limit" in detected_classes:
            self.action = "ACTION: ADJUSTING SPEED TO DETECTED LIMIT"
        elif "traffic_light_green" in detected_classes:
            self.action = "ACTION: GREEN LIGHT DETECTED - CONTINUE DRIVING"
        else:
            self.action = "ACTION: CONTINUE DRIVING"
        return self.action


class TSRDetector:
    CONFIDENCE_THRESHOLD = 0.25

    def __init__(self, model_path, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.vehicle_state = VehicleState()
        self.decision_engine = DecisionEngine()
        self.class_colors = {
            "speed_limit": (255, 0, 0),
            "stop": (0, 0, 255),
            "traffic_light_red": (0, 0, 255),
            "traffic_light_yellow": (0, 255, 255),
            "traffic_light_green": (0, 255, 0)
        }
        
    def reset(self):
        self.vehicle_state = VehicleState()
        self.decision_engine = DecisionEngine()
        
    def process_frame(self, frame):
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        annotated_frame = frame.copy()
        
        detected_classes = []
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()
                cls_name = self.model.names[cls_id]
                detected_classes.append(cls_name)
                
                coordinates = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = map(int, coordinates)
                color = self.class_colors.get(cls_name, (255, 255, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                })
        
        action = self.decision_engine.determine_action(detected_classes)
        stopping_dist = self.vehicle_state.get_stopping_distance()
        speed_limit_detected = "speed_limit" in detected_classes
        
        telemetry = {
            "detected_signs": detected_classes,
            "detections": detections,
            "action": action,
            "speed_kmh": self.vehicle_state.speed_kmh,
            "stopping_distance": stopping_dist,
            "speed_limit_detected": speed_limit_detected,
        }
        

        cv2.putText(annotated_frame, f"Speed: {self.vehicle_state.speed_kmh:.1f} km/h", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Stopping Dist: {stopping_dist:.1f} m", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        

        h, w = frame.shape[:2]
        action_color = (0, 255, 0)
        if "APPLYING BRAKES" in action:
            action_color = (0, 0, 255)
        elif "WARNING" in action or "ADJUSTING" in action:
            action_color = (0, 255, 255)
        cv2.putText(
            annotated_frame,
            action,
            (30, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            action_color,
            3,
        )
        
        return annotated_frame, telemetry
