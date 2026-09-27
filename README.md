# Autonomous Vehicle Perception System

**Master MIATE — Computer Vision Project**

An integrated autonomous-vehicle perception platform combining **9 computer-vision scenarios** into a single Tkinter application.

The system processes video files, images, and live camera input depending on the scenario, producing annotated frames and structured telemetry through a common detector interface.

> **Project status:** Final integrated university prototype
> **Main branch:** Final integrated application
> **Develop branch:** Original team scenario implementations and development history

---

## Overview

The project brings together multiple perception and driver/passenger-safety scenarios under one application.

Instead of each scenario managing its own camera, display loop, and output handling, the final application uses shared components:

```text
                    ┌─────────────────────┐
                    │      Tkinter GUI    │
                    │     Dashboard       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    VideoManager     │
                    │ Video / Image / Cam │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  ScenarioManager    │
                    │ Detector lifecycle  │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       Scenario 1         Scenario 2        Scenario 3...
       Detector           Detector          Detector
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    Annotated Frame
                         + Telemetry
```

Each integrated detector follows the common interface:

```python
annotated_frame, telemetry = detector.process_frame(frame)
```

This allows the GUI to handle input and visualization while each scenario remains responsible for its own perception and decision logic.

---

## Scenarios

| # | Scenario | Main Technology | Input |
|---|---|---|---|
| 1 | Driver Distraction Monitoring | MediaPipe FaceMesh, EAR, head-pose geometry | Video / Camera |
| 2 | Passenger Safety & Posture | MediaPipe Pose, posture heuristics, safety state | Video / Camera |
| 3 | Traffic Sign & Signal Recognition | YOLO-based detection + braking-distance logic | Video |
| 4 | Drivable Area / Free Space | Canny/Hough lane geometry + free-space estimation | Image / Video |
| 5 | Road Surface Conditions | Classical computer vision heuristics | Image / Video / Camera |
| 6 | Construction Zone Detection | HSV color segmentation + cone detection | Image / Video / Camera |
| 7 | Vulnerable Road User Detection | YOLOv8 + ego-corridor reasoning | Video |
| 8 | Vehicle Intent Detection | YOLOv8 + tail-light/blinker analysis | Video |
| 9 | Emergency Vehicle Detection | Visual red/blue flashing-light detection | Video |

---

# Scenario Details

## 1. Driver Distraction Monitoring

Detects potential driver distraction using:

- MediaPipe FaceMesh landmarks
- Head-down score
- Eye Aspect Ratio (EAR)
- Temporal warning state
- Three-second escalation logic
- Simulated speed-reduction / limp-mode state

The system resets the warning state when the driver becomes attentive again.

---

## 2. Passenger Safety & Posture

Monitors passenger posture and safety conditions using MediaPipe Pose.

The scenario includes:

- Leaning detection
- Feet-on-dashboard estimation
- Seatbelt state
- Safety state machine
- Warning and danger states
- Simulated airbag/speed-limit software states

The GUI provides a seatbelt toggle for demonstration purposes.

---

## 3. Traffic Sign & Signal Recognition

Uses a YOLO-based detector for five traffic classes:

- Speed limit
- Stop
- Red light
- Yellow light
- Green light

The scenario combines detection with decision logic and stopping-distance calculations.

The final integrated version does **not** include the original EasyOCR pipeline for reading the numeric value printed on speed-limit signs.

---

## 4. Drivable Area / Free Space

The final integrated implementation uses **classical lane geometry** rather than treating the generic ONNX segmentation output as an active drivable-area class.

The detector uses image-processing techniques such as:

- Edge detection
- Lane-line extraction
- Hough-based geometry
- Lane fitting
- Lane-center estimation
- Free-space/trapezoidal fallback

It produces software-level outputs such as:

- Steer left
- Steer right
- Keep lane

The repository still contains the ONNX model file because it is registered in the project configuration, but it is not the active source of the final drivable-area mask.

---

## 5. Road Surface Conditions

Uses classical computer vision heuristics to estimate road-surface conditions.

The scenario includes detection/estimation related to:

- Dry road
- Wet road
- Snow
- Ice
- Potholes
- Traction-related state
- Simulated maximum-speed recommendations

---

## 6. Construction Zone Detection

Detects construction-zone indicators using computer vision.

Main components include:

- HSV color segmentation
- Orange cone detection
- Cone localization
- Construction-zone state
- Merge/maneuver logic

---

## 7. Vulnerable Road User Detection

Uses YOLOv8 for vulnerable road-user detection.

The system focuses on pedestrians and other relevant road users and combines detection with an ego-vehicle corridor.

The scenario provides software-level warnings such as:

- Pedestrian warning
- Ball/road-user warning
- Automatic-emergency-braking-style decision output

---

## 8. Vehicle Intent Detection

Analyzes nearby vehicles to estimate visual driving intent.

The detector combines:

- YOLO vehicle detection
- Closest/central vehicle selection
- Bounding-box continuity
- Tail-light color analysis
- Blink tracking
- Brake-light detection
- Turn-signal detection
- Hazard-light detection

The output represents an estimated vehicle intent at the perception/software-decision level.

---

## 9. Emergency Vehicle Detection

Detects emergency-vehicle indicators using visual analysis of flashing lights.

The final integrated version focuses on:

- Red-light detection
- Blue-light detection
- Flashing-light behavior
- Emergency alert state
- Pull-over guidance

The final version does **not** use microphone/siren confirmation.

---

# Architecture

The final application is organized into shared infrastructure and scenario-specific detectors.

```text
main.py
│
├── config/
│   └── settings.py
│
├── core/
│   ├── scenario_manager.py
│   └── video_manager.py
│
├── gui/
│   └── dashboard.py
│
├── scenarios/
│   ├── scenario_1/
│   ├── scenario_2/
│   ├── scenario_3/
│   ├── scenario_4/
│   ├── scenario_5/
│   ├── scenario_6/
│   ├── scenario_7/
│   ├── scenario_8/
│   └── scenario_9/
│
├── models/
├── assets/
└── results/
```

## ScenarioManager

`core/scenario_manager.py` manages detector creation and lifecycle.

It:

- Maps scenario keys to detector classes
- Lazily creates detectors
- Supplies model paths
- Stores initialization errors
- Resets detectors
- Releases detector instances

## VideoManager

`core/video_manager.py` centralizes input handling.

It supports:

- Video files
- Images
- Live camera input

It also manages source opening, frame reading, FPS/frame information, progress, and source release.

## Dashboard

`gui/dashboard.py` provides the Tkinter interface.

The dashboard:

- Displays all scenarios
- Groups scenarios by category
- Loads configured demo inputs
- Selects videos/images/cameras
- Requests detectors from `ScenarioManager`
- Processes frames
- Displays annotated frames
- Displays telemetry
- Handles scenario-specific GUI controls
- Reports processing errors

## Settings

`config/settings.py` centralizes:

- Project paths
- Model paths
- Asset paths
- Scenario metadata
- Input modes
- Demo filenames
- GUI configuration

This keeps scenario code independent from the current working directory.

---

# Project Structure

```text
.
├── main.py
├── README.md
├── CONTRIBUTORS.md
├── requirements.txt
├── LICENSE
│
├── config/
│   └── settings.py
│
├── core/
│   ├── scenario_manager.py
│   └── video_manager.py
│
├── gui/
│   └── dashboard.py
│
├── models/
│   ├── yolov8n.pt
│   ├── tsr_best.pt
│   └── yolov8n-seg.onnx
│
├── assets/
│   ├── videos/
│   └── images/
│
├── scenarios/
│   ├── scenario_1/
│   │   └── driver_distraction.py
│   ├── scenario_2/
│   │   └── passenger_safety.py
│   ├── scenario_3/
│   │   └── traffic_sign_recognition.py
│   ├── scenario_4/
│   │   └── drivable_area.py
│   ├── scenario_5/
│   │   └── road_surface.py
│   ├── scenario_6/
│   │   └── construction_detector.py
│   ├── scenario_7/
│   │   └── vru_detector.py
│   ├── scenario_8/
│   │   └── vehicle_intent.py
│   └── scenario_9/
│       └── emergency_vehicle.py
│
├── results/
│
└── docs/
    ├── SCENARIO_CHANGES.md
    ├── DEVELOP_VS_MAIN.md
    └── PRESENTATION_GUIDE.md
```

---

# Models

| Model | Usage |
|---|---|
| `yolov8n.pt` | Vulnerable road user detection and vehicle intent |
| `tsr_best.pt` | Traffic sign and signal recognition |
| `yolov8n-seg.onnx` | Registered for Scenario 4; not used as the active final drivable-area mask |

The project also contains scenarios based on classical computer vision and MediaPipe that do not require additional model files.

---

# Installation & Usage

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Pillow
- PyTorch
- Ultralytics
- ONNX Runtime
- MediaPipe
- Tkinter

## 1. Clone the repository

```bash
git clone git@github.com:moza369/Autonomous-Vehicle-Perception.git
cd Autonomous-Vehicle-Perception
```

## 2. Create a virtual environment

Using a virtual environment is **recommended** to avoid conflicts with Python packages installed on your system.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **If you encounter dependency or package conflicts**, use a fresh virtual environment and reinstall the requirements.

On Debian/Ubuntu systems, if Tkinter is missing:

```bash
sudo apt install python3-tk
```

## 4. Run the application

From the repository root, with the virtual environment activated:

```bash
python main.py
```

The application opens the Tkinter dashboard.

### Basic workflow

1. Launch the application.
2. Select a scenario.
3. Select or use the configured input source.
4. Start the scenario.
5. Observe the annotated frame.
6. Monitor the scenario telemetry.
7. Stop the scenario when finished.

Depending on the scenario, the available inputs can include:

- Video
- Image
- Live camera

Some scenarios have configured demonstration media that can be loaded directly from the application.

## Troubleshooting

### Dependency or import errors

If you encounter errors such as `ModuleNotFoundError`, incompatible package versions, or conflicts with packages already installed on your system, create a clean virtual environment:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

### Tkinter is missing

On Debian/Ubuntu:

```bash
sudo apt install python3-tk
```

Then restart the application.

### Verify the environment

You can check that the main dependencies are available with:

```bash
python -c "import cv2, numpy, PIL, torch, torchvision, ultralytics, onnxruntime, mediapipe; print('All required Python imports: OK')"
```

A successful installation should print:

```text
All required Python imports: OK
```

### Deactivate the virtual environment

When finished:

```bash
deactivate
```


# Example Data Flow

```text
Input
  │
  ▼
VideoManager
  │
  ▼
ScenarioManager
  │
  ▼
Scenario Detector
  │
  ├── Perception
  ├── Scenario logic
  └── Telemetry
  │
  ▼
Annotated Frame + Telemetry
  │
  ▼
Tkinter Dashboard
```

---

# Development Branches

The repository contains two branches with different purposes.

## `main`

Contains the final integrated application.

It provides:

- One GUI
- Shared input management
- Shared detector lifecycle
- Centralized configuration
- Integrated scenario detectors
- Common telemetry interface

## `develop`

Contains the original team scenario implementations and development history.

It remains useful for:

- Reviewing the original implementations
- Comparing scenario logic
- Reviewing original experiments
- Preserving the team's development work

The two branches are intentionally structured differently because they serve different purposes.

See [`docs/DEVELOP_VS_MAIN.md`](docs/DEVELOP_VS_MAIN.md) for the detailed comparison.

---

# Documentation

Additional project documentation:

- [`SCENARIO_CHANGES.md`](docs/SCENARIO_CHANGES.md) — detailed scenario-by-scenario integration changes
- [`DEVELOP_VS_MAIN.md`](docs/DEVELOP_VS_MAIN.md) — architecture and branch comparison
- [`PRESENTATION_GUIDE.md`](docs/PRESENTATION_GUIDE.md) — presentation notes and technical questions for each scenario
- [`CONTRIBUTORS.md`](CONTRIBUTORS.md) — project contributors

---

# Limitations

This project is a **university computer-vision prototype**.

The outputs representing:

- Braking
- Speed reduction
- Steering
- Airbag state
- Lane keeping
- Emergency pull-over
- Automatic emergency braking

are **software-level decisions or simulated actions**.

The application does not directly control a real vehicle ECU, braking system, steering system, airbag system, or other vehicle hardware.

Detection performance also depends on:

- Camera quality
- Lighting conditions
- Input video
- Object visibility
- Model performance
- Scenario-specific heuristics

The system should therefore be considered a perception and decision-making prototype rather than a production autonomous-driving system.

---

# Project Context

This project was developed as part of the **Master MIATE — Computer Vision** coursework.

The objective was to integrate multiple autonomous-vehicle perception scenarios into a unified application while maintaining the individual scenario implementations and development history of the team.

---

# License

See [`LICENSE`](LICENSE) for the project license.
