# Presentation Guide

This guide explains what each original team member should know when presenting
the final integrated application.

## General questions the professor may ask

### Why were the scenarios integrated this way?

The original scenarios had different input loops, output methods, and model
paths. The final application needs one GUI and one processing lifecycle.
Detector classes and shared managers provide a consistent way to select,
process, display, and reset scenarios.

### Why use detector classes?

A detector class keeps scenario state between frames while exposing a common
`process_frame(frame)` method. This is required for temporal logic such as
distraction duration, blink histories, emergency-light alternation, and
vehicle tracking.

### Why use `process_frame()`?

The dashboard and `VideoManager` own the input loop. A detector should process
one already-read frame and return its annotated result and telemetry instead of
opening its own camera or video window.

### Why centralize `VideoManager`?

It avoids nine different implementations of video opening, camera opening,
frame reading, FPS handling, image loading, and resource release.

### Why centralize model paths?

`config/settings.py` and `models/` provide stable paths independent of the
current working directory. `ScenarioManager` passes those paths to detectors.

### How does telemetry work?

Each detector returns a dictionary with scenario-specific values. The
dashboard iterates over the dictionary and displays the values in the
telemetry panel.

### How does the GUI communicate with detectors?

The dashboard requests a detector from `ScenarioManager`, reads frames through
`VideoManager`, calls `process_frame(frame)`, and displays the returned frame
and telemetry.

### Which components are simulations rather than real vehicle control?

The action messages, speed values, braking decisions, airbag state, lane
keeping state, and pull-over arrows are software perception/decision outputs.
The repository does not contain vehicle ECU, brake, steering, airbag, or
cruise-control interfaces.

---

# Scenario 1 — Driver Distraction

## What the original team member developed

The original project used MediaPipe FaceMesh, head-down geometry, and Eye
Aspect Ratio. Its standalone webcam loop escalated from a warning to simulated
limp mode after three seconds.

## What remains in the final version

- MediaPipe FaceMesh.
- Head-down score.
- EAR calculation.
- Warning status.
- Three-second escalation.
- Simulated speed reduction.

## What changed during integration

The detector function and webcam loop became
`DriverDistractionDetector.process_frame()`. The dashboard and
`VideoManager` now own input and display. Telemetry was added.

## Professor questions and short answers

1. **What measurements indicate distraction?**  
   Head-down score and average EAR.

2. **What is the warning duration?**  
   Three seconds of continuous warning.

3. **What happens after three seconds?**  
   Limp mode becomes true and the simulated speed decreases toward 30 km/h.

4. **What happens when the driver becomes attentive?**  
   The timer and limp mode reset.

5. **Does it detect phones?**  
   No, not in the final code.

## Technical concepts to revise

- MediaPipe FaceMesh landmarks.
- Eye Aspect Ratio.
- Normalized landmark coordinates.
- Temporal state and reset behavior.
- Difference between perception output and vehicle control.

## Do not claim

- Direct phone detection.
- Physical cruise-control disengagement.
- Real speed reduction.

---

# Scenario 2 — Passenger Safety / Posture

## What the original team member developed

The original project separated MediaPipe posture detection, seatbelt state,
safety state-machine logic, audio alarm, and webcam presentation.

## What remains in the final version

- Leaning detection.
- Feet-on-dashboard posture check.
- Seatbelt state.
- Safety state.
- Airbag and speed-limit software state.
- Warning and danger behavior.

## What changed during integration

The separate files became `PassengerSafetyDetector`. The GUI now provides the
seatbelt toggle, and `VideoManager` supplies frames. Structured telemetry and
exact warning/action messages were added.

## Professor questions and short answers

1. **How is leaning detected?**  
   Nose position is compared with shoulder-center position.

2. **How are feet on the dashboard estimated?**  
   Visible ankle height is compared with hip-center height.

3. **What triggers the combined warning?**  
   The passenger is both unbuckled and out of position.

4. **How is the seatbelt tested?**  
   Through the dashboard’s software toggle.

5. **Does the system control a real airbag?**  
   No, it changes an internal state and displays an action.

## Technical concepts to revise

- MediaPipe Pose landmarks.
- Geometric posture heuristics.
- State-machine transitions.
- GUI-to-detector state synchronization.
- Non-blocking warning behavior.

## Do not claim

- Physical airbag disabling.
- Complete detection of all unsafe postures.
- A real seatbelt sensor connection.

---

# Scenario 3 — Traffic Sign and Signal Recognition

## What the original team member developed

The original TSR project used YOLO, a vehicle-state model, decision logic,
stopping distance, HUD rendering, output video, and optional EasyOCR for
numeric speed-limit reading.

## What remains in the final version

- YOLO traffic class detection.
- The five traffic classes.
- Decision priority.
- Stopping-distance calculation.
- Annotated detections.
- Telemetry.

## What changed during integration

The large notebook/script workflow became `TSRDetector`. The model path is
centralized. The final confidence threshold is `0.25` instead of the original
`0.35`. The reviewed final detector does not contain the original EasyOCR
numeric speed-limit pipeline.

## Professor questions and short answers

1. **What classes are detected?**  
   Speed limit, stop, red light, yellow light, and green light.

2. **What has the highest priority?**  
   Stop sign and red-light braking take priority over speed or green signals.

3. **How is stopping distance computed?**  
   Reaction distance plus braking distance using speed, reaction time, friction,
   and gravity.

4. **Can the final code read the number on a speed-limit sign?**  
   Not in the reviewed final detector; the original EasyOCR code is absent.

5. **What confidence threshold is used?**  
   `0.25`.

## Technical concepts to revise

- YOLO bounding boxes and confidence.
- Detection class priority.
- Braking-distance equations.
- Model-path configuration.
- Difference between class recognition and OCR.

## Do not claim

- Numeric speed-limit OCR in the final code.
- Physical emergency braking.
- That changing the threshold has a documented reason; the repository does not
  state one.

---

# Scenario 4 — Drivable Area / Free Space

## What the original team member developed

The original project loaded a YOLO segmentation ONNX model, selected a
BDD100K image, interpreted an output mask, calculated the mask center, and
displayed a steering decision.

## What remains in the final version

- Drivable/free-space overlay concept.
- Lane-center offset concept.
- Steering-left, steering-right, and keep-lane messages.
- Central ONNX model path registration.

## What changed during integration

The final detector does not use the generic ONNX output as the drivable-area
mask. It uses Canny edges, Hough lines, lane fitting, a trapezoid mask, and a
fallback corridor. It is now video-capable and returns telemetry.

## Professor questions and short answers

1. **Why is the ONNX mask not used?**  
   The available model is generic COCO segmentation and has no drivable-area
   class.

2. **How are lanes found?**  
   Canny edges and probabilistic Hough lines are filtered and fitted.

3. **How is steering direction selected?**  
   Lane center is compared with image center.

4. **What if lane markings are unavailable?**  
   A centered trapezoid is used as fallback.

5. **Is confidence a neural-network probability?**  
   No, it is a heuristic based on line evidence.

## Technical concepts to revise

- Canny edge detection.
- Hough line segments.
- Lane-line slope filtering.
- Free-space polygons.
- Heuristic confidence.

## Do not claim

- Trained drivable-area segmentation in the final active algorithm.
- Pixel-perfect road segmentation.
- Real steering control.

---

# Scenario 5 — Road Surface Conditions

## What the original team member developed

The original project was a multi-file classical-CV pipeline for wet, snow,
ice, dry-road, and pothole conditions. It supported image, video, and webcam
modes with optional debug output and saved results.

## What remains in the final version

- Road-region extraction.
- Brightness, saturation, darkness, reflection, and edge features.
- Wet/snow/ice heuristic classification.
- Pothole contour detection.
- Traction and maximum-speed decisions.

## What changed during integration

The helper modules were combined into `RoadSurfaceDetector`. Input, display,
and output writing moved into shared components. The detector now returns
telemetry instead of owning a CLI.

## Professor questions and short answers

1. **How is wetness estimated?**  
   Dark ratio, reflection ratio, and brightness variation contribute to a wet
   score.

2. **How are snow and ice separated?**  
   The code uses different combinations of brightness, white ratio, saturation,
   and edge density.

3. **How are potholes detected?**  
   Dark and edge-connected candidates are filtered by contour geometry.

4. **Is a trained model used?**  
   No, the final scenario uses classical CV.

5. **Does maximum speed control the vehicle?**  
   No, it is a reported decision value.

## Technical concepts to revise

- HSV and grayscale features.
- Morphological closing/opening.
- Canny edges.
- Contour filtering.
- Rule-based classification.

## Do not claim

- A deep-learning weather classifier.
- Guaranteed performance under all lighting/weather.
- Actual traction-control or speed-control commands.

---

# Scenario 6 — Construction Zone

## What the original team member developed

The original project detected orange cones with HSV segmentation, contour
filtering, cone counting, and a fitted path line. Three or more cones produced
the construction-zone decision.

## What remains in the final version

- HSV orange detection.
- Blur and morphology.
- Cone contour geometry.
- Three-cone construction-zone threshold.
- Fitted line.
- Merge-left overlay.

## What changed during integration

The detector moved from `construction_zone/` to
`scenarios/scenario_6/`. The public API became `process_frame()`. The minimum
cone area changed from 300 to 320, and final telemetry is simpler.

## Professor questions and short answers

1. **Why HSV?**  
   It makes orange hue thresholding practical.

2. **What triggers the construction-zone state?**  
   At least three accepted cone detections.

3. **How are false positives reduced?**  
   Morphology, area, dimensions, aspect ratio, and top-region filtering.

4. **Is a trained detector used?**  
   No, the final implementation is classical CV.

5. **Does it really disable lane keeping?**  
   No, it reports a software action.

## Technical concepts to revise

- HSV masks.
- Morphological operations.
- Contour geometry.
- `cv2.fitLine`.
- Threshold sensitivity.

## Do not claim

- Concrete-barrier recognition.
- Physical lane-keeping deactivation.
- A complete construction-zone understanding system.

---

# Scenario 7 — Vulnerable Road User Detection

## What the original team member developed

The original project used YOLOv8 nano, a trapezoidal ego corridor, COCO class
IDs, pedestrian emergency braking, sports-ball warnings, and a cooldown timer.

## What remains in the final version

- YOLOv8 nano.
- Corridor geometry.
- Pedestrian and sports-ball handling.
- Emergency-braking and warning messages.
- Temporal action stability.

## What changed during integration

The fixed-video loop and output writer became `VRUDetector.process_frame()`.
The shared model path and dashboard now control input and display.

## Professor questions and short answers

1. **How is the predicted vehicle path represented?**  
   By a trapezoidal polygon.

2. **How is an object tested against it?**  
   The bottom-center point of its bounding box is tested.

3. **Why does a pedestrian trigger braking?**  
   A pedestrian in the corridor is treated as a collision risk.

4. **Why does a ball trigger a warning?**  
   The final logic gives sports balls a warning/pre-charge response.

5. **Does the code brake the real vehicle?**  
   No, it reports and displays the action.

## Technical concepts to revise

- YOLO COCO classes.
- Bounding-box geometry.
- Point-in-polygon testing.
- Ego-path approximation.
- Cooldown/state timing.

## Do not claim

- Physical emergency braking.
- Complete coverage of all animals, cyclists, and motorcyclists.
- A fully autonomous collision-avoidance controller.

---

# Scenario 8 — Vehicle Intent Detection

## What the original team member developed

The original project detected the closest centered vehicle, tracked identity
with IoU, analyzed red and amber tail-light regions, detected braking and
turning, synchronized hazards, and stabilized messages over time.

## What remains in the final version

- YOLO vehicle detection.
- Centered closest-vehicle selection.
- IoU continuity.
- Tail-light crop and left/right split.
- Adaptive blink tracking.
- Brake and hazard decisions.
- Intent telemetry.

## What changed during integration

The original video-processing function became `VehicleIntentDetector`.
Scenario-specific CLI, video writing, and display moved into shared components.
The model path is centralized at `models/yolov8n.pt`.

## Professor questions and short answers

1. **How is the lead vehicle selected?**  
   Centered car, bus, or truck candidates are scored by area and center offset.

2. **Why use IoU?**  
   To reduce switching between different vehicles.

3. **How are turn signals detected?**  
   Left and right light regions are analyzed over temporal ON/OFF histories.

4. **How are hazards identified?**  
   Both sides blink and their histories meet the synchronization requirement.

5. **Does image-left always equal vehicle-left?**  
   Only after camera mirroring and viewpoint are verified.

## Technical concepts to revise

- YOLO filtering.
- IoU.
- HSV color segmentation.
- Temporal hysteresis/debounce.
- Blink synchronization.

## Do not claim

- Perfect performance in every dark or compressed video.
- Guaranteed physical left/right without camera calibration.
- Real braking or steering control.

---

# Scenario 9 — Emergency Vehicle Detection

## What the original team member developed

The original project detected red/blue flashing lights and also used
`sounddevice` microphone input and RMS volume to estimate siren presence.
Emergency detection required visual flashing and siren detection.

## What remains in the final version

- HSV red mask.
- HSV blue mask.
- Temporal red/blue alternation.
- Colored boxes.
- Emergency alert.
- Pull-over arrow.
- Action telemetry.

## What changed during integration

The microphone stream, audio callback, volume state, siren threshold, and
audio-dependent confirmation were removed. The final detector is visual-only
and is called through the common frame interface.

## Professor questions and short answers

1. **What visual signal is detected?**  
   Alternating red and blue emergency-light regions.

2. **Does the final system hear the siren?**  
   No, microphone and audio code are absent.

3. **How is flashing measured?**  
   Dominant color states are stored and transitions are counted.

4. **What does the pull-over action do?**  
   It displays an alert and arrow.

5. **Does the vehicle actually pull over?**  
   No, there is no control interface.

## Technical concepts to revise

- HSV thresholding.
- Red hue wraparound.
- Temporal state history.
- False positives from colored lights.
- Difference between perception and actuation.

## Do not claim

- Siren recognition.
- Microphone support.
- Automatic physical pull-over.

---

## Final presentation reminder

Each student should distinguish between:

1. Detection performed by computer vision.
2. A decision represented in telemetry or an overlay.
3. Real vehicle control, which is not implemented in this repository.

The final application is an integrated perception and decision-making
demonstration. Its action messages are software outputs for the university
prototype; they are not direct commands to a production vehicle.
