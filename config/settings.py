"""Central configuration for the Autonomous Driving Perception System."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

YOLOV8N_MODEL = os.path.join(MODELS_DIR, "yolov8n.pt")
TSR_MODEL = os.path.join(MODELS_DIR, "tsr_best.pt")
DRIVABLE_AREA_MODEL = os.path.join(MODELS_DIR, "yolov8n-seg.onnx")

for d in [MODELS_DIR, ASSETS_DIR, VIDEOS_DIR, IMAGES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

SCENARIOS = {
    "driver_distraction": {
        "name": "Driver Distraction Monitoring",
        "category": "DRIVER & PASSENGER SAFETY",
        "description": "Monitors driver attention using facial landmark analysis. Detects head pitch and eye closure to identify distraction. Triggers cruise control disengagement and speed reduction after 3 seconds of inattention.",
        "input_modes": ["video", "live_camera"],
        "default_input": "live_camera",
        "model": None,  # Uses MediaPipe built-in
        "alerts": [
            "WARNING: EYES ON ROAD",
            "ACTION: DISENGAGING CRUISE CONTROL - SLOWING DOWN",
        ],
    },
    "passenger_safety": {
        "name": "Passenger Posture & Seatbelt",
        "category": "DRIVER & PASSENGER SAFETY",
        "description": "Monitors passenger posture and seatbelt status using pose estimation. Detects forward/sideways leaning and feet on dashboard. Controls airbag deployment and speed limiting based on safety state.",
        "input_modes": ["video", "live_camera"],
        "default_input": "live_camera",
        "model": None,  # Uses MediaPipe built-in
        "alerts": [
            "WARNING: PASSENGER UNBUCKLED & OUT OF POSITION",
            "ACTION: PASSENGER AIRBAG DISABLED - SPEED RESTRICTED",
        ],
    },
    "traffic_sign_recognition": {
        "name": "Traffic Sign & Signal Recognition",
        "category": "TRAFFIC UNDERSTANDING",
        "description": "Detects traffic signs (speed limit, stop) and traffic lights (red, yellow, green) using a custom-trained YOLO model. Computes physical braking distances and displays a Tesla-style HUD.",
        "input_modes": ["video"],
        "default_input": "video",
        "model": "tsr_best.pt",
        "demo_video": "scenario_3_Traffic_sign.mp4",
        "alerts": [
            "ACTION: RED LIGHT DETECTED - APPLYING BRAKES",
            "ACTION: STOP SIGN DETECTED - APPLYING BRAKES",
            "WARNING: YELLOW LIGHT DETECTED - PREPARE TO STOP",
            "ACTION: ADJUSTING SPEED TO DETECTED LIMIT",
            "ACTION: GREEN LIGHT DETECTED - CONTINUE DRIVING",
        ],
    },
    "drivable_area": {
        "name": "Drivable Area Detection",
        "category": "ROAD & ENVIRONMENT PERCEPTION",
        "description": "Segments the drivable road area using ONNX-based YOLOv8 segmentation. Computes lane center offset and provides steering correction recommendations.",
        "input_modes": ["image", "video"],
        "default_input": "image",
        "model": "yolov8n-seg.onnx",
        "alerts": ["ACTION: STEERING LEFT", "ACTION: STEERING RIGHT", "ACTION: KEEP LANE"],
    },
    "road_surface": {
        "name": "Road Surface Conditions",
        "category": "ROAD & ENVIRONMENT PERCEPTION",
        "description": "Detects road surface conditions (dry, wet, snow, ice) and large potholes using classical computer vision. Adjusts traction control mode and maximum safe speed.",
        "input_modes": ["image", "video", "live_camera"],
        "default_input": "video",
        "model": None,  # Classical CV only
        "demo_video": "scenario_5_Road_surface_condition.mp4",
        "alerts": ["WARNING: WET ROAD", "WARNING: LARGE POTHOLE"],
    },
    "vru_detection": {
        "name": "Vulnerable Road User Detection",
        "category": "VULNERABLE ROAD USERS",
        "description": "Detects pedestrians and other vulnerable road users in the vehicle's forward trajectory. Triggers autonomous emergency braking (AEB) when a VRU enters the ego corridor.",
        "input_modes": ["video"],
        "default_input": "video",
        "model": "yolov8n.pt",
        "demo_video": "scenario_7_Vulnerable_road.mp4",
        "alerts": ["ACTION: EMERGENCY AUTOMATIC BRAKING", "WARNING: OBJECT IN PATH"],
    },
    "vehicle_intent": {
        "name": "Vehicle Intent & Tail Lights",
        "category": "TRAFFIC UNDERSTANDING",
        "description": "Tracks the lead vehicle's tail lights to detect braking, turn signals, and hazard lights. Uses HSV color analysis and temporal blink tracking for reliable intent classification.",
        "input_modes": ["video"],
        "default_input": "video",
        "model": "yolov8n.pt",
        "demo_video": "scenario_8_vehicle_intent.mp4",
        "alerts": ["ACTION: LEAD VEHICLE BRAKING", "INFO: TURN SIGNAL"],
    },
    "emergency_vehicle": {
        "name": "Emergency Vehicle Detection",
        "category": "TRAFFIC UNDERSTANDING",
        "description": "Detects approaching emergency vehicles by analyzing flashing blue/red light patterns. Triggers pull-over maneuver when confirmed.",
        "input_modes": ["video"],
        "default_input": "video",
        "model": None,  # Classical CV
        "demo_video": "scenario_9_Emergency_Vehicle.mp4",
        "alerts": ["ACTION: PULLING OVER TO RIGHT SHOULDER"],
    },
    "construction_zone": {
        "name": "Construction Zone Detection",
        "category": "ROAD & ENVIRONMENT PERCEPTION",
        "description": "Detects construction zones by identifying orange traffic cones using HSV color segmentation. Disables lane keeping assist and initiates merge-left maneuver.",
        "input_modes": ["image", "video", "live_camera"],
        "default_input": "video",
        "model": None,  # Classical CV only
        "demo_video": "scenario_6_construction_zone.mp4",
        "alerts": ["ACTION: CONSTRUCTION ZONE - MERGING LEFT"],
    },
}

GUI_TITLE = "Autonomous Vehicle Perception System"
GUI_SUBTITLE = "Master MIATE — Computer Vision Integrated Platform"
GUI_WIDTH = 1400
GUI_HEIGHT = 900

COLORS = {
    "bg_dark": "#0d1117",
    "bg_medium": "#161b22",
    "bg_card": "#21262d",
    "bg_card_hover": "#30363d",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "accent_blue": "#58a6ff",
    "accent_green": "#3fb950",
    "accent_orange": "#d29922",
    "accent_red": "#f85149",
    "accent_purple": "#bc8cff",
    "border": "#30363d",
    "status_ready": "#3fb950",
    "status_running": "#58a6ff",
    "status_warning": "#d29922",
    "status_error": "#f85149",
}
