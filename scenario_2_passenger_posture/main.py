import cv2
from pose_detector import PoseDetector
from seatbelt_detector import SeatbeltDetector
from state_machine import SafetySystem
from audio import Alarm

# Initialisation du detecteur et du systeme de securite
pose_detector = PoseDetector()
seatbelt_detector = SeatbeltDetector()
safety_system = SafetySystem(current_speed=80)
alarm = Alarm()

cap = cv2.VideoCapture(0)
vehicle_speed = 80

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Effet miroir pour la camera
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Analyse de posture et de ceinture
    posture_info = pose_detector.detect(frame)
    seatbelt = seatbelt_detector.get_status()

    # Mise a jour de la machine a etats
    state = safety_system.update(seatbelt, posture_info)
    airbag = safety_system.get_airbag_status()
    speed_limit = safety_system.get_speed_limit()

    seatbelt_off = (seatbelt == "OFF")
    out_of_position = posture_info["out_of_position"]

    # Choix de la couleur selon l'etat du systeme
    if state == "NORMAL":
        color = (0, 255, 0)
    elif state == "WARNING":
        color = (0, 255, 255)
        alarm.play()
    else:
        color = (0, 0, 255)
        alarm.play()

    # Affichage des informations sur l'ecran (HUD)
    cv2.putText(frame, "Occupant Safety Monitoring", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicle Speed : {vehicle_speed} km/h", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    seatbelt_color = (0, 255, 0) if seatbelt == "ON" else (0, 0, 255)
    cv2.putText(frame, f"Seatbelt : {seatbelt}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, seatbelt_color, 2)

    posture_color = (0, 255, 0) if posture_info["label"] == "SAFE" else (0, 0, 255)
    cv2.putText(frame, f"Posture : {posture_info['label']}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)

    airbag_color = (0, 255, 0) if airbag == "ENABLED" else (0, 0, 255)
    cv2.putText(frame, f"Airbag : {airbag}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, airbag_color, 2)

    speed_txt = f"Speed Limit : {speed_limit} km/h" if speed_limit else "Speed Limit : None"
    cv2.putText(frame, speed_txt, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"System State : {state}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Affichage des bandeaux d'alertes en fonction de la situation
    if seatbelt_off and out_of_position:
        # Alerte critique
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 110), (w, h - 60), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "WARNING: PASSENGER UNBUCKLED & OUT OF POSITION", (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Action requise
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 55), (w, h - 5), (0, 120, 200), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"ACTION: PASSENGER AIRBAG DISABLED - SPEED RESTRICTED TO {vehicle_speed} KM/H", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    elif seatbelt_off:
        # Alerte ceinture
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 70), (w, h - 20), (0, 180, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "WARNING: PASSENGER UNBUCKLED", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif out_of_position:
        # Alerte posture
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 70), (w, h - 20), (0, 120, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"WARNING: PASSENGER {posture_info['label'].upper()} - AIRBAG DISABLED", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Raccourcis clavier pour tester
    cv2.putText(frame, "Press S : Toggle Seatbelt | ESC : Exit", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Occupant Safety Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        seatbelt_detector.toggle()
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()