# Autonomous Vehicle Perception System — Integrated Platform

**Master MIATE — Computer Vision**

A unified Tkinter GUI that integrates **9 autonomous-driving perception scenarios** into one application.  
Each scenario processes **your own video, image, or live camera feed** and displays annotated results with real-time telemetry.

---

## Quick Start

```bash
cd integrated_project
pip install -r requirements.txt
python main.py
```

> **Note:** Tkinter is usually bundled with Python. If missing: `sudo apt-get install python3-tk`

---

## Scenarios

| Category | Module | Description | Input |
|---|---|---|---|
| **Driver & Passenger Safety** | Driver Distraction Monitoring | MediaPipe FaceMesh head pitch & eye closure detection | Video / Camera |
| | Passenger Posture & Seatbelt | MediaPipe Pose for posture + seatbelt toggle | Video / Camera |
| **Road & Environment** | Drivable Area Detection | ONNX YOLOv8-seg lane segmentation & steering | Image / Video |
| | Road Surface Conditions | Classical CV: wet, snow, ice, pothole detection | Image / Video / Camera |
| | Construction Zone Detection | HSV orange cone detection & merge maneuver | Image / Video / Camera |
| **Traffic Understanding** | Traffic Sign & Signal Recognition | Custom YOLO11n (5 classes) + physics HUD | Video |
| | Emergency Vehicle Detection | HSV flashing red/blue lights | Video |
| | Vehicle Intent & Tail Lights | HSV tail light blink tracking & braking detection | Video |
| **Vulnerable Road Users** | VRU Detection | YOLOv8n pedestrian/cyclist AEB system | Video |

---

## How to Use

1. **Launch** `python main.py`
2. **Select a scenario** from the sidebar (organized by category)
3. **Choose input:**
   - **Select Video** — pick any `.mp4` / `.avi` file
   - **Select Image** — pick any `.jpg` / `.png` file
   - **Live Camera** — uses webcam
   - Scenarios with demo videos auto-load them (you can still override)
4. **Click "Run Scenario"** to start processing
5. **View results** in the video canvas with live telemetry on the right panel
6. **Click "Stop"** to halt processing

> For **Passenger Posture**, use the **"Toggle Seatbelt"** button to simulate seatbelt on/off.

---

## Project Structure

```
integrated_project/
├── main.py                          # Entry point — launches the GUI
├── requirements.txt                 # All dependencies
├── README.md                        # This file
│
├── config/
│   └── settings.py                  # Central configuration, scenario metadata, colors
│
├── core/
│   ├── scenario_manager.py          # Lazy-loads scenario detectors
│   └── video_manager.py             # Video/image/camera input handler
│
├── gui/
│   └── dashboard.py                 # Tkinter dashboard with sidebar + video display
│
├── scenarios/
│   ├── scenario_1/
│   │   └── driver_distraction.py    # MediaPipe FaceMesh driver attention
│   ├── scenario_2/
│   │   └── passenger_safety.py      # MediaPipe Pose + seatbelt state machine
│   ├── scenario_3/
│   │   └── traffic_sign_recognition.py  # YOLO TSR + physics decision engine
│   ├── scenario_4/
│   │   └── drivable_area.py         # ONNX segmentation + lane center
│   ├── scenario_5/
│   │   └── road_surface.py          # Classical CV surface + pothole detection
│   ├── scenario_7/
│   │   └── vru_detector.py          # YOLO VRU + ego-corridor AEB
│   ├── scenario_8/
│   │   └── vehicle_intent.py        # HSV blink tracker + braking detection
│   ├── scenario_9/
│   │   └── emergency_vehicle.py     # HSV flashing red/blue lights
│   ├── scenario_6/
│   │   └── construction_detector.py # HSV cone detection + merge logic
│
├── models/
│   ├── yolov8n.pt                   # Shared YOLOv8 nano (VRU + Vehicle Intent)
│   ├── tsr_best.pt                  # Custom YOLO11n for traffic signs (5 classes)
│   └── yolov8n-seg.onnx             # ONNX segmentation for drivable area
│
├── assets/
│   ├── videos/                      # Demo videos (auto-loaded per scenario)
│   └── images/                      # Demo images for road surface / drivable area
│
└── results/                         # Output directory
```

---

## Models

| Model | Size | Used By | Source |
|---|---|---|---|
| `yolov8n.pt` | 6.5 MB | VRU Detection, Vehicle Intent | Pre-trained YOLOv8 nano |
| `tsr_best.pt` | 5.2 MB | Traffic Sign Recognition | Custom fine-tuned YOLO11n |
| `yolov8n-seg.onnx` | 105.8 MB | Drivable Area Detection | YOLOv8n-seg exported to ONNX |

Scenarios 1, 2, 5, 9, and Construction Zone use **no external model files** (MediaPipe built-in or classical CV only).

---

## Dependencies

- Python 3.10+
- OpenCV, NumPy, Pillow
- PyTorch + Ultralytics (YOLO scenarios)
- ONNX Runtime (drivable area)
- MediaPipe (driver distraction, passenger posture)
- Tkinter (usually included with Python)

---

## Important Notes

- This integrated project is a **completely separate copy**. The original team scenario folders remain **untouched**.
- All paths are **relative** — the project is portable.
- Each scenario adapter is **self-contained** — no imports from original scenario directories.
- Demo videos are stored under `assets/videos/` and auto-load when you select a scenario.
