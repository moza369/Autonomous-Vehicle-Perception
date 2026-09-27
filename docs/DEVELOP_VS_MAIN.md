# `origin/develop` vs `main`

## Comparison references

This document compares:

- `origin/develop`: `c0b42fdbab97edadf85708e415f37bb482431644`
- `main` / final integrated commit: `b1556928c68777b3ff3e2544b5bcf950e07ef839`

The main branch commit is titled:

```text
Integrate autonomous vehicle perception system
```

## Purpose of develop

`develop` contains the original team scenario implementations. The projects
were organized around independent ownership and independent testing. Each
scenario could have its own:

- Entry point.
- Requirements file.
- Input source.
- Model path.
- OpenCV display loop.
- Output writer.
- Results directory.
- Helper modules.

The scenario directories have different naming and coding styles because they
were developed separately:

```text
scenario_1_driver_distraction/
scenario_2_passenger_posture/
scenario_3_TSR/
scenario_4_drivable_area/
scenario_5_road_surface/
construction_zone/
scenario_7_vulnerable_road_user_detection/
vehicle_intent_detection/
scenario9_Emergency Vehicle Detection/
```

## Purpose of main

`main` is the final integrated application. It provides one GUI and one
application lifecycle for all nine scenarios:

```text
main.py
config/
core/
gui/
models/
assets/
results/
scenarios/
requirements.txt
```

The goal of this structure is consistent scenario loading, input handling,
frame processing, annotation display, telemetry display, and model/asset
path management.

## Original project structure

The original branch has scenario-local implementations such as:

- Scenario 1: `detector.py` plus webcam `main.py`.
- Scenario 2: separate pose, seatbelt, state-machine, audio, and main files.
- Scenario 3: a notebook and a large `test_model.py`.
- Scenario 4: dataset/model/results folders and a single `src/main.py`.
- Scenario 5: multiple classical-CV helper modules and a CLI main.
- Scenario 6: a detector and standalone main.
- Scenario 7: separate offline/live entry points.
- Scenario 8: one large video-processing script.
- Scenario 9: one script with visual and microphone logic.

## Final project structure

The final branch contains:

```text
main.py
config/
    settings.py
core/
    scenario_manager.py
    video_manager.py
gui/
    dashboard.py
models/
    yolov8n.pt
    tsr_best.pt
    yolov8n-seg.onnx
assets/
    videos/
    images/
results/
scenarios/
    scenario_1/driver_distraction.py
    scenario_2/passenger_safety.py
    scenario_3/traffic_sign_recognition.py
    scenario_4/drivable_area.py
    scenario_5/road_surface.py
    scenario_6/construction_detector.py
    scenario_7/vru_detector.py
    scenario_8/vehicle_intent.py
    scenario_9/emergency_vehicle.py
```

Each scenario has an `__init__.py` and one primary detector module.

## Why the structures differ

The branches serve different purposes:

- `develop` supports independent team development and scenario-specific
  experimentation.
- `main` supports one executable application with common infrastructure.

Therefore, code was not always removed because its algorithm was unnecessary.
Some code was refactored into detector classes, while input/display/lifecycle
responsibilities were moved into common modules.

There are also real behavioral changes. The most important verified examples
are:

- Scenario 3 no longer contains the original EasyOCR numeric speed-limit
  pipeline in its final detector.
- Scenario 4 no longer uses the ONNX output as the active drivable-area mask;
  it uses lane geometry.
- Scenario 9 no longer uses microphone/siren confirmation.

## How scenarios were integrated

The final scenario contract is effectively:

```python
annotated_frame, telemetry = detector.process_frame(frame)
```

The integration process converted scenario-specific loops into reusable
objects. State that used to be local to a `while` loop became detector state.
Scenario-specific `cv2.imshow()` and video-writing behavior moved to the
dashboard/application layer.

The exact final behavior differs by scenario, so the individual comparisons
in `SCENARIO_CHANGES.md` remain the source of truth for algorithmic changes.

## ScenarioManager

`core/scenario_manager.py` is the detector factory and lifecycle manager.

It maps scenario keys to detector classes:

```text
driver_distraction       → DriverDistractionDetector
passenger_safety         → PassengerSafetyDetector
traffic_sign_recognition → TSRDetector
drivable_area             → DrivableAreaDetector
road_surface              → RoadSurfaceDetector
vru_detection             → VRUDetector
vehicle_intent            → VehicleIntentDetector
emergency_vehicle        → EmergencyVehicleDetector
construction_zone        → ConstructionZoneDetector
```

It also:

- Lazily creates detectors.
- Supplies model paths from `config.settings`.
- Stores initialization errors.
- Resets detectors before a new run.
- Releases detector instances.

## VideoManager

`core/video_manager.py` centralizes media input. It supports:

- Video files.
- Live camera input.
- Single images.

It validates paths and open operations, reads frames, tracks FPS and frame
count, exposes progress, and releases the active source.

This replaces scenario-specific ownership of `cv2.VideoCapture()` in the
integrated execution path.

## Dashboard

`gui/dashboard.py` provides the Tkinter application. It:

- Displays all scenarios from `SCENARIOS`.
- Groups scenario buttons by category.
- Loads configured demo videos.
- Allows video, image, and camera selection where supported.
- Requests a detector from `ScenarioManager`.
- Resets the detector.
- Opens the input through `VideoManager`.
- Calls `process_frame()` for each frame.
- Displays annotated frames.
- Displays telemetry.
- Handles the passenger-safety seatbelt toggle.
- Reports initialization and frame-processing errors.

## Settings and configuration

`config/settings.py` centralizes:

- `BASE_DIR`.
- `MODELS_DIR`.
- `ASSETS_DIR`.
- `VIDEOS_DIR`.
- `IMAGES_DIR`.
- `RESULTS_DIR`.
- Model paths.
- Scenario names and categories.
- Descriptions.
- Input modes.
- Default input modes.
- Demo video names.
- GUI dimensions and colors.

This avoids scenario code depending on the current working directory or
scenario-local model paths.

## Shared models

The final branch stores shared model files under `models/`:

```text
models/yolov8n.pt
models/tsr_best.pt
models/yolov8n-seg.onnx
```

The model usage is not identical across scenarios:

- `yolov8n.pt` is used by VRU and vehicle-intent detectors.
- `tsr_best.pt` is used by traffic-sign recognition.
- `yolov8n-seg.onnx` is registered for Scenario 4, but the final Scenario 4
  detector uses lane geometry rather than treating its generic segmentation
  output as a drivable-area class.

## Shared assets

The final application centralizes media under:

```text
assets/videos/
assets/images/
```

The dashboard obtains default demo filenames from `config/settings.py`.
Scenario-local standalone media and output loops are not required by the
integrated GUI.

## Detector interface

The common integration idea is:

```python
detector = ScenarioManager().get_detector(key)
annotated, telemetry = detector.process_frame(frame)
```

The exact telemetry keys differ by scenario because each scenario reports
different information. The dashboard accepts the returned dictionary and
renders its values.

## End-to-end data flow

```text
Dashboard
    ↓ selects scenario and source
VideoManager
    ↓ opens source and reads frame
ScenarioManager
    ↓ creates/resets selected detector
Scenario Detector
    ↓ process_frame(frame)
annotated_frame + telemetry
    ↓
Dashboard display and telemetry panel
```

## What a developer should modify

### When working with original team logic

Use the original directories on `develop` to study:

- Original algorithms.
- Original helper modules.
- Original training/model assumptions.
- Original demos.
- Original standalone behavior.

### When modifying the final application

Use `main` and the integrated directories:

- Detector algorithm: `scenarios/scenario_N/`.
- Detector creation: `core/scenario_manager.py`.
- Input behavior: `core/video_manager.py`.
- GUI behavior: `gui/dashboard.py`.
- Paths, metadata, and demo filenames: `config/settings.py`.
- Models: `models/`.
- Demo media: `assets/videos/`.

Changes involving a new scenario, new input mode, new model, new telemetry
field, or new GUI control may require updates in more than one of these
locations.

This document describes the architecture; it does not recommend merging one
branch into the other.
