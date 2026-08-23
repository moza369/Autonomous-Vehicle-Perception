import cv2
import numpy as np
from ultralytics import YOLO

def main():
    
    model = YOLO("models/yolov8n.pt") 
    
    
    video_path = "videos/test_driving_video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    PEDESTRIAN_CLASS = 0
    SPORTS_BALL_CLASS = 32
    OTHER_VRU_CLASSES = [1, 3, 15, 16, 17, 18, 21, 22]

   
    trajectory_poly = np.array([
        [int(width * 0.12), int(height * 0.95)], 
        [int(width * 0.42), int(height * 0.55)], 
        [int(width * 0.58), int(height * 0.55)], 
        [int(width * 0.88), int(height * 0.95)]  
    ], np.int32)

   
    COOLDOWN_FRAMES = 12  
    brake_timer = 0
    warning_timer = 0

    print("\n=======================================================")
    print(" LIVE TRACKING ACTIVE: Press 'q' on the window to exit.")
    print("=======================================================\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream reached the end frame.")
            break

      
        results = model(frame, verbose=False)[0]
        
        pedestrian_in_lane = False
        ball_in_lane = False

        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [trajectory_poly], (255, 255, 0))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [trajectory_poly], isClosed=True, color=(255, 255, 0), thickness=2)

        for box in results.boxes:
            class_id = int(box.cls[0])
            
            if class_id == PEDESTRIAN_CLASS or class_id == SPORTS_BALL_CLASS or class_id in OTHER_VRU_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bottom_center_pt = (int((x1 + x2) / 2), y2)
                
               
                is_inside = cv2.pointPolygonTest(trajectory_poly, bottom_center_pt, False)

                if is_inside >= 0: 
                    if class_id == PEDESTRIAN_CLASS:
                        pedestrian_in_lane = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.circle(frame, bottom_center_pt, 6, (0, 0, 255), -1)
                    elif class_id == SPORTS_BALL_CLASS:
                        ball_in_lane = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        
        if pedestrian_in_lane:
            brake_timer = COOLDOWN_FRAMES  
            warning_timer = 0 
        elif ball_in_lane:
            if brake_timer == 0:          
                warning_timer = COOLDOWN_FRAMES
        else:
            if brake_timer > 0:   brake_timer -= 1
            if warning_timer > 0: warning_timer -= 1

        
        if brake_timer > 0:
            msg = "ACTION: PEDESTRIAN DETECTED - EMERGENCY AUTOMATIC BRAKING"
            cv2.rectangle(frame, (0, height - 90), (width, height), (0, 0, 255), cv2.FILLED) 
            cv2.putText(frame, msg, (35, height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        elif warning_timer > 0:
            msg = "WARNING: OBJECT IN PATH - PRE-CHARGING BRAKE ACTUATORS"
            cv2.rectangle(frame, (0, height - 90), (width, height), (0, 165, 255), cv2.FILLED) 
            cv2.putText(frame, msg, (35, height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (0, height - 90), (width, height), (30, 30, 30), cv2.FILLED) 
            cv2.putText(frame, "SYSTEM STATUS: PATH CLEAR - AUTONOMOUS CRUISING", (35, height - 38), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        
        cv2.imshow("MIATE - Live Autonomous System Perception Feed", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("System interrupted by user exit.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
