# Scenario Changes: `origin/develop` → `main`

## Integration architecture

The `develop` branch contains independent team projects. Each project commonly
owns its input loop, model path, display window, output writer, and command-line
entry point. The final `main` branch wraps the scenarios in one application:

```text
Dashboard
    ↓
VideoManager
    ↓
ScenarioManager
    ↓
Scenario Detector
    ↓
annotated_frame + telemetry
    ↓
Dashboard
```

`core/video_manager.py` opens videos, images, and cameras and reads frames.
`core/scenario_manager.py` creates and resets detector instances. The GUI in
`gui/dashboard.py` selects scenarios and inputs, calls `process_frame(frame)`,
displays the annotated frame, and displays telemetry. Paths, model locations,
scenario metadata, and default demo videos are centralized in
`config/settings.py`.

The comparison uses `origin/develop` at commit
`c0b42fdbab97edadf85708e415f37bb482431644` and `main` at commit
`b1556928c68777b3ff3e2544b5bcf950e07ef839`.

---

# Scenario 1 — Driver Distraction

## Original implementation

The original implementation was located in:

```text
scenario_1_driver_distraction/
├── config.py
├── detector.py
├── main.py
└── requirements.txt
```

`detector.py` used MediaPipe FaceMesh to calculate:

- A head-down score from the forehead, nose, chin, and eye landmarks.
- Eye Aspect Ratio (EAR) for both eyes.
- `WARNING: EYES ON ROAD` when the head-down or eye-closure condition was met.

`main.py` opened a webcam directly, maintained the distraction timer, displayed
the warning, activated limp mode after three seconds, and reduced the simulated
speed from 60 km/h toward 30 km/h.

## Final implementation

The final implementation is:

```text
scenarios/scenario_1/driver_distraction.py
```

It provides `DriverDistractionDetector`, which owns MediaPipe FaceMesh,
distraction timing, simulated speed, limp-mode state, annotation, and
telemetry. `ScenarioManager` creates it, and `VideoManager` or the dashboard
supplies frames.

## What changed

### Architecture

- The standalone `detect_driver_attention()` function and webcam loop became
  the class `DriverDistractionDetector`.
- State that was previously held in `main.py` is now instance state.
- `process_frame()` and `reset()` were added for the common detector contract.

### Algorithm

The core FaceMesh, head-down, and EAR calculations are present in the final
source. The final constants are:

```text
DOWN_SCORE_THRESHOLD = 0.24
EAR_THRESHOLD = 0.20
WARNING_DURATION_LIMIT = 3.0 seconds
INITIAL_SPEED = 60.0 km/h
MIN_SPEED = 30.0 km/h
SPEED_DECREASE_STEP = 0.2
```

The final action is combined into one line:

```text
ACTION: DISENGAGING CRUISE CONTROL - SLOWING DOWN
```

### Models

No trained external model was added. Both versions use MediaPipe FaceMesh.

### Input/output

- Direct webcam ownership moved from the scenario to `VideoManager`.
- Direct `cv2.imshow()` control moved to the dashboard.
- The final detector returns an annotated frame and a telemetry dictionary.

### GUI/integration

`ScenarioManager` constructs `DriverDistractionDetector()`. The dashboard
shows the warning, action, duration, speed, and other telemetry.

### Removed functionality

The standalone webcam application and its independent window loop are not in
the final scenario module. This behavior was moved to shared application
components rather than removed from the overall application.

### Added functionality

- Detector lifecycle through `reset()`.
- Structured telemetry.
- Unified video, image, and camera input.
- Dashboard presentation.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Entry point | Scenario-specific webcam `main.py` | Unified dashboard |
| Detector | `detect_driver_attention()` | `DriverDistractionDetector` |
| Detection | FaceMesh, head-down score, EAR | Same core calculations |
| Escalation | Main-loop state | Detector instance state |
| Output | OpenCV window | Frame plus telemetry in GUI |
| Model | MediaPipe FaceMesh | MediaPipe FaceMesh |

## What the original team member should know

The main FaceMesh detection work remains recognizable. It was refactored into
a reusable detector class. The final application does not directly control a
vehicle; speed reduction and cruise-control disengagement are simulated
software outputs.

## Professor questions

1. **How is distraction detected?**  
   The final code uses a head-down score and average eye aspect ratio.

2. **What activates limp mode?**  
   Continuous `WARNING: EYES ON ROAD` status for at least three seconds.

3. **What happens when attention returns?**  
   The timer and limp mode reset, and simulated speed returns to 60 km/h.

4. **Does the system detect phone usage?**  
   No. The final code detects facial distraction indicators, not a phone.

5. **Does it control a real vehicle?**  
   No. It updates an internal speed value and displays an action.

## Important limitations

- Do not claim direct phone detection.
- Do not claim real cruise-control or vehicle-speed control.
- The code does not implement a physical steering, braking, or ECU interface.

---

# Scenario 2 — Passenger Safety / Posture

## Original implementation

The original implementation was:

```text
scenario_2_passenger_posture/
├── audio.py
├── main.py
├── pose_detector.py
├── seatbelt_detector.py
├── state_machine.py
└── requirements.txt
```

The original `PoseDetector` used MediaPipe Pose. It classified leaning by
comparing the nose with the shoulder center and classified feet-on-dashboard
when visible ankles were above the hip center. `SeatbeltDetector` supplied
seatbelt state. `SafetySystem` managed normal, warning, and danger states,
including airbag status and speed limit. `main.py` opened the webcam, flipped
the frame, displayed the HUD, and used `Alarm`.

## Final implementation

The final implementation is:

```text
scenarios/scenario_2/passenger_safety.py
```

It combines pose handling, seatbelt state, safety decisions, overlays,
telemetry, reset behavior, and the optional warning chime in one detector.

## What changed

### Architecture

The original pose detector, seatbelt detector, state machine, alarm usage, and
main loop were adapted into `PassengerSafetyDetector`. The detector exposes
`process_frame()` and `reset()`.

### Algorithm

The final code retains the original posture concepts:

- Nose versus shoulder-center geometry for leaning.
- Ankle versus hip geometry for feet-on-dashboard.
- Seatbelt state combined with posture state.

The final warning/action strings include:

```text
WARNING: PASSENGER UNBUCKLED & OUT OF POSITION
ACTION: PASSENGER AIRBAG DISABLED - SPEED RESTRICTED TO <current speed> KM/H
```

### Models

No external trained model was introduced. The final implementation continues
to use MediaPipe-based posture processing.

### Input/output

- Webcam ownership moved to `VideoManager`.
- The dashboard supplies video, image, or camera frames.
- The final detector returns annotations and telemetry rather than owning an
  OpenCV loop.

### GUI/integration

`dashboard.py` has a passenger-safety-specific seatbelt button and synchronizes
the selected seatbelt state with the detector. `ScenarioManager` creates
`PassengerSafetyDetector()`.

### Removed functionality

- Standalone webcam entry point.
- Keyboard-only seatbelt control inside the scenario loop.
- Direct scenario-owned OpenCV window.

These responsibilities moved to the shared dashboard and video manager.

### Added functionality

- Unified telemetry.
- GUI seatbelt toggle.
- Detector reset/lifecycle handling.
- Exact combined warning/action messages.
- Current-speed restriction display.
- Non-blocking optional warning chime behavior.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Structure | Multiple helper files and `main.py` | One integrated detector module |
| Pose logic | MediaPipe Pose | Same posture concepts |
| Seatbelt control | Keyboard toggle | Dashboard button |
| Input | Direct webcam | Shared `VideoManager` |
| Output | OpenCV HUD and alarm | GUI frame and telemetry |
| Vehicle action | Internal airbag/speed state | Internal state and dashboard message |

## What the original team member should know

The original posture and seatbelt concepts were combined into the unified
detector. The final airbag and speed restriction are software states and
messages. The code does not communicate with a real airbag controller or
vehicle ECU.

## Professor questions

1. **How is leaning detected?**  
   The final code compares nose position with shoulder-center position.

2. **How are feet on the dashboard detected?**  
   Visible ankle positions are compared with the hip-center position.

3. **What happens when the passenger is both unbuckled and unsafe?**  
   The combined warning and airbag/speed-restriction action are displayed.

4. **How is the seatbelt tested in the GUI?**  
   The dashboard includes a seatbelt toggle that synchronizes detector state.

5. **Does the code physically disable the airbag?**  
   No. It changes an internal safety state and reports the action.

## Important limitations

- Do not claim physical airbag control.
- Do not claim that the pose model identifies every possible unsafe posture.
- The seatbelt interaction is a software simulation in the GUI.

---

# Scenario 3 — Traffic Sign and Signal Recognition

## Original implementation

The original implementation was located in:

```text
scenario_3_TSR/
├── TSR_Traffic_Sign_Signal_Recognition.ipynb
├── test_model.py
├── Poids/
├── Resultats/
└── requirements.txt
```

The original `test_model.py` used Ultralytics YOLO and defined the classes:

```text
speed_limit
stop
traffic_light_red
traffic_light_yellow
traffic_light_green
```

It included `VehicleState`, `DecisionEngine`, stopping-distance calculation,
HUD rendering, output-video writing, and optional EasyOCR for reading the
numeric speed-limit value. It used confidence `0.35`, IoU `0.45`, and image
size 640.

## Final implementation

The final implementation is:

```text
scenarios/scenario_3/traffic_sign_recognition.py
```

It contains `VehicleState`, `DecisionEngine`, and `TSRDetector`. It loads the
central model path `models/tsr_best.pt`, filters detections at confidence
`0.25`, draws boxes, computes stopping distance, decides an action, and
returns telemetry.

## What changed

### Architecture

The notebook/script pipeline became `TSRDetector.process_frame()`. Model
construction is performed by `ScenarioManager`.

### Algorithm

The final decision priority is:

1. Red light.
2. Stop sign.
3. Yellow light.
4. Speed-limit sign.
5. Green light.
6. Continue driving.

The final action messages are explicit, for example:

```text
ACTION: RED LIGHT DETECTED - APPLYING BRAKES
ACTION: STOP SIGN DETECTED - APPLYING BRAKES
WARNING: YELLOW LIGHT DETECTED - PREPARE TO STOP
```

The final source uses confidence threshold `0.25`, while the original
`test_model.py` used `0.35`.

The final detector reviewed on `main` does not contain the original EasyOCR
numeric speed-limit reading pipeline.

### Models

The model path changed from scenario-local `best.pt` usage to:

```text
models/tsr_best.pt
```

The exact training-history difference is not determinable from the available
code/history.

### Input/output

- Fixed script input and output paths were replaced with shared input handling.
- The detector returns an annotated frame and telemetry.
- The final detector does not write an annotated output video itself.

### GUI/integration

`ScenarioManager` creates `TSRDetector(model_path=...)`. The dashboard displays
the annotated frame, action, detections, speed, and stopping distance.

### Removed functionality

From the final scenario module, the following develop functionality is absent:

- EasyOCR speed-limit value extraction.
- OCR caching every five frames.
- Script-level output-video writer.
- The standalone CUDA/CPU selection workflow.
- Notebook/script execution flow.

These functions are not relocated into `ScenarioManager`, `VideoManager`, or
the dashboard. Numeric OCR is absent from the reviewed final detector.

### Added functionality

- Detector class with reset support.
- Per-frame telemetry.
- Central model path.
- Safe native-integer box coordinates.
- Unified GUI integration.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Model | Scenario-local `best.pt` | Shared `models/tsr_best.pt` |
| Confidence | 0.35 | 0.25 |
| Speed sign | Optional EasyOCR numeric reading | Generic `speed_limit` class decision |
| Execution | Notebook/script and output writer | Frame detector and GUI |
| Decisions | Vehicle state and stopping distance | Vehicle state and stopping distance |
| Output | HUD and video file | Frame plus telemetry |

## What the original team member should know

The core YOLO class recognition and decision priorities remain, but the final
implementation is smaller. The original EasyOCR capability is not present in
the final detector. The final system can identify a speed-limit sign class but
does not, in the reviewed code, reliably extract its printed number.

## Professor questions

1. **Which classes are detected?**  
   Speed-limit signs, stop signs, and red, yellow, and green traffic lights.

2. **What has priority if a red light and speed-limit sign appear together?**  
   The decision engine prioritizes the red light and requests braking.

3. **How is stopping distance computed?**  
   Reaction distance and braking distance are calculated from speed, reaction
   time, friction, and gravity.

4. **Can the final detector read 30, 50, or 80 from a sign?**  
   The reviewed final detector identifies `speed_limit` but does not include
   the original EasyOCR numeric-reading code.

5. **What confidence threshold is used?**  
   `TSRDetector.CONFIDENCE_THRESHOLD` is `0.25`.

## Important limitations

- Do not claim numeric speed-limit OCR in the final implementation.
- Do not claim physically applied brakes.
- The exact reason for changing the confidence threshold is not documented.

---

# Scenario 4 — Drivable Area / Free Space

## Original implementation

The original implementation was:

```text
scenario_4_drivable_area/
├── datasets/
├── models/
├── results/
├── src/main.py
└── requirements.txt
```

`src/main.py` randomly selected a BDD100K image, loaded
`yolov8n-seg.onnx` through ONNX Runtime, used `outputs[1][0]` as a mask,
thresholded it, created a colored overlay, calculated the mean mask x-position,
displayed a steering action, and saved one result image.

## Final implementation

The final implementation is:

```text
scenarios/scenario_4/drivable_area.py
```

It contains `DrivableAreaDetector`. The final detector uses lane geometry:

- Canny edges.
- Probabilistic Hough lines.
- Left/right candidate filtering.
- Lane-line fitting.
- A trapezoidal free-space mask.
- A centered fallback when lane evidence is unavailable.

It returns lane center, image center, offset, confidence, status, and action
telemetry.

## What changed

### Architecture

The random single-image script became a reusable detector with
`process_frame()` and internal lane-estimation methods.

### Algorithm

This is a verified algorithm change. The final active algorithm does not use
the ONNX output as a drivable-area mask. It estimates a road corridor from
lane-marking geometry.

The final steering mapping is:

- Lane center right of image center → steering left.
- Lane center left of image center → steering right.
- Small offset → keep lane.

### Models

The model path is centralized as `models/yolov8n-seg.onnx`, but the final
detector documentation states that this is a generic COCO segmentation model
without a drivable-area class. The active final detection method is therefore
classical lane geometry.

### Input/output

- Random dataset-image selection was removed.
- Single saved-image workflow was replaced by frame processing.
- Video and GUI input are supported through shared components.
- Output includes confidence and telemetry.

### GUI/integration

`ScenarioManager` constructs `DrivableAreaDetector(model_path=...)`, while
`VideoManager` and the dashboard provide and display frames.

### Removed functionality

- Random BDD100K image selection.
- ONNX Runtime mask processing as the active algorithm.
- Direct image saving and direct OpenCV window ownership.

These were replaced by the integrated frame pipeline.

### Added functionality

- Hough-based lane estimation.
- Left/right line fitting.
- Free-space trapezoid fallback.
- Confidence telemetry.
- Video processing.
- OpenCV-shape-compatible Hough segment iteration.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Input | Random BDD100K image | Video/image frame from GUI |
| Active algorithm | ONNX output mask | Canny/Hough lane geometry |
| Center estimate | Mean mask pixels | Lane-boundary midpoint |
| Missing evidence | No documented fallback | Centered trapezoid fallback |
| Output | Saved image | Annotated frame and telemetry |
| Model | ONNX Runtime mask | Path retained for compatibility; not active for road mask |

## What the original team member should know

This scenario underwent the largest algorithmic change. The final implementation
should be presented as a lane/free-space estimation proof of concept, not as
the original trained drivable-area segmentation pipeline.

## Professor questions

1. **Why is the ONNX model not used for the final drivable-area mask?**  
   The available model is documented as generic COCO segmentation and has no
   drivable-area class.

2. **How are lane markings extracted?**  
   Canny edges and probabilistic Hough line segments are filtered by slope and
   position.

3. **How is steering direction selected?**  
   The fitted lane center is compared to the image center.

4. **What happens if lane evidence is missing?**  
   A centered trapezoidal road corridor is used as a fallback.

5. **Is confidence a trained probability?**  
   No. It is a heuristic based on detected lane lines and Hough segments.

## Important limitations

- Do not claim trained drivable-area segmentation in the final detector.
- Do not claim pixel-accurate road segmentation.
- Do not claim real steering actuation.
- The confidence value is heuristic.

---

# Scenario 5 — Road Surface Conditions

## Original implementation

The original implementation was:

```text
scenario_5_road_surface/
├── main.py
├── preprocessing.py
├── road_surface_detector.py
├── pothole_detector.py
├── utils.py
├── visualization.py
├── input/
└── output/
```

It supported images, videos, and webcam input. The processing pipeline
selected a road region, classified dry/wet/snow/ice conditions, detected large
potholes, decided traction and speed actions, annotated frames, and optionally
wrote output images/videos. It also had CLI arguments and debug windows.

## Final implementation

The final implementation is:

```text
scenarios/scenario_5/road_surface.py
```

It combines the main helper functionality into one module and exposes
`RoadSurfaceDetector.process_frame()`.

It retains classical feature-based classification and pothole detection using
brightness, saturation, reflections, edges, masks, contours, aspect ratio,
area, location, and circularity.

## What changed

### Architecture

The original helper modules became data classes and functions inside one
detector module.

### Algorithm

The reviewed final source retains the original classical-CV approach:

- Road-region extraction.
- Normalization and contrast preprocessing.
- Surface feature calculation.
- Wet/snow/ice heuristic scores.
- Dark-edge pothole candidates.
- Morphological filtering.
- Rule-based vehicle action.

### Models

No trained model is used.

### Input/output

- CLI image/video/camera handling moved to shared components.
- Output-video writing moved outside the detector.
- Telemetry is returned with the annotated frame.

### GUI/integration

`ScenarioManager` creates `RoadSurfaceDetector()`. The dashboard displays
condition, confidence, pothole count, warning, action, traction mode, and
maximum speed.

### Removed functionality

- Standalone CLI parser.
- Direct output-video writer.
- Scenario-owned OpenCV windows.
- Optional debug-window workflow from the standalone script.

### Added functionality

- Common detector lifecycle.
- Structured telemetry.
- Dashboard display.
- Centralized input handling.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Structure | Six-module pipeline plus CLI | One detector module |
| Algorithm | Classical CV | Same general classical CV |
| Input | CLI image/video/webcam | Shared manager and GUI |
| Output | Saved files and windows | Frame plus telemetry |
| Model | None | None |

## What the original team member should know

The detection pipeline is mostly refactored rather than replaced. The final
version is easier to call from the GUI, but the classifications remain
heuristic classical-CV decisions.

## Professor questions

1. **How is a wet road identified?**  
   The classifier combines dark-pixel ratio, reflection ratio, and brightness
   variation.

2. **How are snow and ice separated?**  
   They use different combinations of white ratio, brightness, saturation, and
   edge density.

3. **How are potholes detected?**  
   Dark candidate regions are combined with edge regions and filtered by
   geometry and contour properties.

4. **Does it use deep learning?**  
   No. The final scenario uses classical computer vision.

5. **Does maximum speed control the vehicle?**  
   No. It is a decision output displayed in telemetry.

## Important limitations

- Do not claim a trained road-condition model.
- Do not claim guaranteed performance across weather, cameras, or roads.
- Do not claim real traction-control or speed-control actuation.

---

# Scenario 6 — Construction Zone

## Original implementation

The original implementation was:

```text
construction_zone/
├── detector.py
├── main.py
├── test_images/
├── test_videos/
└── requirements.txt
```

`ConstructionZoneDetector` used HSV orange segmentation, Gaussian blur,
morphological cleanup, contour filtering, cone geometry, and a minimum of
three detected cones to identify a construction zone. It fitted a line through
cone ground centers and produced merge-left metadata.

The original minimum cone area was `300`.

## Final implementation

The final implementation is:

```text
scenarios/scenario_6/construction_detector.py
```

The final detector continues to use HSV cone detection, fitted cone path
geometry, and the merge-left overlay. The final minimum cone area is `320`,
and frames are resized to 640×480 before processing.

## What changed

### Architecture

Original methods such as `detect_cones()`, `analyze_scene()`,
`draw_path_line()`, and `process_image()` were adapted into the common
`process_frame()` API and internal helper methods.

### Algorithm

The core cone algorithm is preserved. The documented threshold change is:

```text
Original minimum area: 300
Final minimum area: 320
```

### Models

No trained model is used.

### Input/output

- Direct scenario-owned input moved to `VideoManager`.
- Scenario-specific output writing moved out of the detector.
- Detailed original metadata was simplified into final telemetry.

### GUI/integration

`ScenarioManager` imports:

```python
scenarios.scenario_6.construction_detector
```

The dashboard displays the annotated frame and action telemetry.

### Removed functionality

The final telemetry does not expose every original metadata field, including
the original cone-coordinate list, fitted-line tuple, and slope field.
Standalone window and output handling were moved to shared components.

### Added functionality

- Common detector interface.
- Unified manager and dashboard integration.
- Central asset/input handling.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Location | `construction_zone/` | `scenarios/scenario_6/` |
| Detection | HSV cone detection | Same core method |
| Minimum area | 300 | 320 |
| Input | Standalone script | VideoManager/dashboard |
| Output | Detailed metadata | Simplified telemetry |
| Model | None | None |

## What the original team member should know

The main orange-cone algorithm remains. The final implementation is not a
trained construction-zone model and does not independently understand every
type of work-zone barrier.

## Professor questions

1. **Why use HSV?**  
   HSV allows an explicit orange hue range with saturation/value filtering.

2. **What triggers a construction zone?**  
   At least three valid cone detections.

3. **How are false positives reduced?**  
   Blur, morphology, top-region exclusion, contour area, dimensions, and aspect
   ratio are used.

4. **Are concrete barriers detected by a separate model?**  
   Not in the final detector reviewed here; the implementation is cone-focused.

5. **Does it physically disable lane keeping?**  
   No. It reports and displays a software action.

## Important limitations

- Do not claim a trained barrier detector.
- Do not claim actual lane-keeping control.
- Do not claim the final telemetry contains every original metadata field.

---

# Scenario 7 — Vulnerable Road User Detection

## Original implementation

The original implementation was:

```text
scenario_7_vulnerable_road_user_detection/
├── src/main.py
├── src/main_live.py
├── models/
├── videos/
└── results/
```

The original video processor loaded YOLOv8 nano, created a trapezoidal ego
corridor, treated pedestrian class `0` as a braking trigger, treated sports
ball class `32` as a warning object, and used a cooldown timer. It wrote an
annotated output video.

## Final implementation

The final implementation is:

```text
scenarios/scenario_7/vru_detector.py
```

It uses YOLOv8 nano, a trapezoidal path corridor, class filtering, pedestrian
emergency braking, ball warning, annotations, telemetry, and state timing.

## What changed

### Architecture

The fixed-video loop became `VRUDetector.process_frame()`. The detector is
created by `ScenarioManager`.

### Algorithm

The corridor and class-based decision concepts are retained. The bottom center
of each bounding box is tested against the corridor.

### Models

The model path moved to shared:

```text
models/yolov8n.pt
```

The model family and COCO class basis remain the same.

### Input/output

- Fixed video and direct output writer were replaced by `VideoManager`.
- Annotated frames and telemetry are returned to the dashboard.

### GUI/integration

The dashboard controls source selection and display. `ScenarioManager` supplies
the shared model path.

### Removed functionality

- Standalone fixed-video loop.
- Direct output-video writing.
- Scenario-owned OpenCV window.

### Added functionality

- Detector lifecycle.
- Reset handling.
- Structured telemetry.
- Unified GUI operation.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Model | Local `models/yolov8n.pt` | Shared `models/yolov8n.pt` |
| Input | Fixed video | Shared video/image/camera interface |
| Corridor | Trapezoid | Trapezoid |
| Pedestrian action | Emergency braking | Emergency braking |
| Ball action | Warning | Warning |
| Output | Saved video | Frame and telemetry |

## What the original team member should know

The main VRU logic remains in the final detector. The primary change is the
execution architecture: the detector no longer owns the video loop or output
writer.

## Professor questions

1. **How is the ego path represented?**  
   A trapezoidal polygon is used as the expected vehicle trajectory.

2. **How is corridor membership checked?**  
   The bottom-center point of the detection box is tested with
   `pointPolygonTest`.

3. **Why does a pedestrian cause braking?**  
   A pedestrian in the vehicle corridor is treated as an immediate collision
   risk.

4. **Why is a ball handled differently?**  
   The final logic uses a warning/pre-charge response for the sports-ball class.

5. **Does the code perform real emergency braking?**  
   No. It displays and reports the action.

## Important limitations

- Do not claim a physical brake command.
- Do not claim that every vulnerable-road-user category is equally validated.
- Corridor geometry is a simplified proof-of-concept model.

---

# Scenario 8 — Vehicle Intent Detection

## Original implementation

The original implementation was:

```text
vehicle_intent_detection/
├── vehicle_intent_detection.py
├── videos/
├── README.md
└── requirements.txt
```

It used YOLOv8 nano to select the closest centered vehicle, IoU for identity
continuity, HSV tail-light analysis, adaptive left/right blink trackers,
braking detection, hazard synchronization, lost-frame reset, stable messages,
and optional output-video writing.

## Final implementation

The final implementation is:

```text
scenarios/scenario_8/vehicle_intent.py
```

It contains the equivalent detector architecture in a class:

- `_get_closest_vehicle()`
- `_iou()`
- `_get_taillight_region()`
- `_get_red_intensity()`
- `_get_signal_intensity()`
- `_BlinkTracker`
- `VehicleIntentDetector`

It returns intent, braking, turn-signal, hazard, and status telemetry.

## What changed

### Architecture

The original `process_video()` loop and local state became
`VehicleIntentDetector.process_frame()` and instance state.

### Algorithm

The main tracking and blink logic is present in the final source:

- YOLO vehicle classes `[2, 5, 7]`.
- Vehicle selection by area and center offset.
- IoU continuity.
- Tail-light crop.
- Red braking analysis.
- Left/right signal analysis.
- Blink history.
- Hazard synchronization.
- Message stability.

The local working-tree history previously included low-light indicator
adjustments, but the exact relationship between local uncommitted changes and
remote `main` cannot be determined from GitHub branch history alone.

### Models

Both versions use YOLOv8 nano and COCO vehicle classes. The model path is now
centralized at `models/yolov8n.pt`.

### Input/output

- CLI source/output options were replaced by shared input handling.
- The detector no longer writes the output video itself.
- The final result is an annotated frame plus telemetry.

### GUI/integration

`ScenarioManager` constructs `VehicleIntentDetector(model_path=...)`. The
dashboard renders the detector output and telemetry.

### Removed functionality

- Standalone command-line processing.
- Direct output-video writer.
- Scenario-owned OpenCV loop.

These responsibilities moved to shared components.

### Added functionality

- Reusable detector class.
- Reset lifecycle.
- Structured intent telemetry.
- Unified GUI integration.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Detector | Standalone video function | `VehicleIntentDetector` |
| Tracking | Closest vehicle and IoU | Same core logic |
| Signals | Blink trackers and HSV | Same core logic in class |
| Input | CLI video/webcam | Shared VideoManager/dashboard |
| Output | Optional output video | Frame and telemetry |
| Model | YOLOv8 nano | Shared YOLOv8 nano |

## What the original team member should know

This scenario was largely refactored rather than algorithmically replaced.
Explain the lead-vehicle selection, identity continuity, temporal blink
tracking, and synchronized hazards. Physical left/right interpretation depends
on camera orientation and mirroring.

## Professor questions

1. **How is the lead vehicle chosen?**  
   Centered car, bus, or truck candidates are scored using area and center
   offset.

2. **Why is IoU used?**  
   It helps ensure that tracking does not switch to a different nearby vehicle.

3. **How are hazards detected?**  
   Both side trackers must blink and their histories must be synchronized.

4. **Why use temporal history?**  
   A turn indicator is a blinking signal, not merely a single bright pixel.

5. **Can image-left always be called vehicle-left?**  
   Not without knowing whether the camera image is mirrored.

## Important limitations

- Do not claim perfect physical left/right interpretation without camera
  orientation verification.
- Do not claim direct vehicle control.
- Do not claim that every lighting condition is equally robust.

---

# Scenario 9 — Emergency Vehicle Detection

## Original implementation

The original implementation was:

```text
scenario9_Emergency Vehicle Detection/
├── scenario 9.py
├── ambulance.mp4
├── README.md
└── requirements.txt
```

It used HSV red/blue masks and temporal alternation history. It also opened a
`sounddevice` microphone stream, calculated RMS volume, estimated siren
presence, and required both flashing lights and siren detection before
triggering the emergency alert.

## Final implementation

The final implementation is:

```text
scenarios/scenario_9/emergency_vehicle.py
```

It is a visual-only `EmergencyVehicleDetector` using red/blue HSV masks,
temporal alternation counting, colored bounding boxes, an emergency overlay,
a pull-over arrow, and telemetry.

## What changed

### Architecture

The fixed-video top-level loop became `process_frame()`. State is held by the
detector instance and created by `ScenarioManager`.

### Algorithm

Original confirmation logic:

```text
flashing lights AND siren
```

Final confirmation logic:

```text
flashing lights
```

The visual red/blue alternation logic remains, but audio confirmation is absent.

### Models

No trained model is used.

### Input/output

- Fixed `ambulance.mp4` handling moved to `VideoManager` and the dashboard.
- The detector returns annotated frames and telemetry instead of owning a
  playback loop.

### GUI/integration

`ScenarioManager` creates `EmergencyVehicleDetector()` without audio
parameters. Configuration describes visual red/blue flashing lights.

### Removed functionality

The following is absent from the final source:

- `sounddevice`.
- Microphone input stream.
- Audio callback.
- RMS volume tracking.
- Siren threshold.
- Audio-dependent confirmation.
- Audio-stream cleanup.

This functionality was removed, not relocated to the shared video or GUI
components.

### Added functionality

- Class-based detector.
- Shared input handling.
- Structured telemetry.
- Visual-only detection contract.

## Original → Final

| Aspect | Original | Final |
|---|---|---|
| Visual detection | HSV red/blue alternation | HSV red/blue alternation |
| Audio | Microphone siren check | Absent |
| Trigger | Lights and siren | Visual flashing lights |
| Input | Fixed video | Shared manager/dashboard |
| Output | OpenCV window | Frame and telemetry |
| Model | None | None |

## What the original team member should know

The visual part of the original work remains, but microphone functionality was
removed. The final scenario must be presented as visual-only emergency-light
detection.

## Professor questions

1. **What identifies an emergency vehicle?**  
   Alternating red and blue light regions detected in HSV space.

2. **Does the final code detect sirens?**  
   No. Audio code is absent from the final implementation.

3. **How is flashing identified?**  
   Dominant red/blue states are stored and transitions are counted over time.

4. **What does the pull-over action do?**  
   It displays an action message and visual arrow.

5. **Does the system physically pull over?**  
   No. There is no vehicle-control interface.

## Important limitations

- Do not claim siren or microphone recognition in the final version.
- Do not claim automatic physical pull-over.
- Color-based flashing detection can be affected by lighting and unrelated
  red/blue objects.

---

# Overall conclusions

- Scenarios 1, 5, 6, 7, and 8 retain substantial portions of their original
  detection logic but are refactored into detector classes.
- Scenario 3 is simplified relative to the original TSR/OCR prototype.
- Scenario 4 changes its active algorithm from ONNX mask interpretation to
  classical lane geometry.
- Scenario 9 removes audio confirmation and becomes visual-only.
- The largest common change is the movement of input, display, state lifecycle,
  paths, and telemetry into shared integration components.
