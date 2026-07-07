# Scenario 1 - Driver Distraction & Attention Detection

## Objective

This module implements Scenario 1 of the Autonomous Vehicle Perception project.

The goal is to detect whether the driver is distracted while driving. In this scenario, the driver looks down at a phone while the vehicle is moving. The system must warn the driver and simulate a safety action if the distraction continues for more than 3 seconds.

## Scenario Description

The vehicle is assumed to be moving at 60 km/h.

If the driver looks down, the system displays:

```text
WARNING: EYES ON ROAD
```

If the driver does not correct their attention after 3 seconds, the system displays:

```text
ACTION: DISENGAGING CRUISE CONTROL
SLOWING DOWN
```

The system also simulates a progressive reduction of the vehicle speed.

## Used Technologies

### OpenCV

OpenCV is used to:

- open the webcam;
- read video frames in real time;
- draw text and visual information on the video;
- display the output window;
- manage keyboard input such as pressing `q` to quit.

### MediaPipe Face Mesh

MediaPipe Face Mesh is used to detect facial landmarks on the driver's face.

These landmarks allow the system to analyze:

- head position;
- eye position;
- nose position;
- chin position;
- eye openness.

### NumPy

NumPy is used for numerical calculations, especially to compute distances between facial landmarks.

## Detection Method

The system combines two main indicators.

### 1. Head Down Score

The head down score estimates whether the driver's head is tilted downward.

It uses the following facial landmarks:

- forehead;
- nose;
- chin;
- left eye;
- right eye.

If the score is higher than the threshold defined in `config.py`, the system considers that the driver may be looking down.

### 2. Eye Aspect Ratio

The Eye Aspect Ratio, also called EAR, measures how open the eyes are.

If the EAR value is too low, this may indicate that the eyes are nearly closed or not properly directed toward the road.

## Decision Rule

The driver is considered distracted if at least one of the following conditions is true:

```text
down_score > DOWN_SCORE_THRESHOLD
```

or:

```text
average_ear < EAR_THRESHOLD
```

If one of these conditions is true, the system displays:

```text
WARNING: EYES ON ROAD
```

If the warning continues for more than 3 seconds, the system displays:

```text
ACTION: DISENGAGING CRUISE CONTROL
SLOWING DOWN
```

## Project Structure

```text
scenario_1_driver_distraction/
│
├── config.py
├── detector.py
├── main.py
├── requirements.txt
├── README.md
└── outputs/
```

## File Description

### config.py

This file contains the configuration values used by the system:

- webcam index;
- head-down threshold;
- EAR threshold;
- warning duration limit;
- initial simulated speed;
- minimum simulated speed;
- speed decrease step.

### detector.py

This file contains the computer vision logic:

- MediaPipe Face Mesh initialization;
- facial landmark extraction;
- head-down score calculation;
- Eye Aspect Ratio calculation;
- driver attention classification;
- frame annotation.

### main.py

This file runs the complete real-time system:

- opens the webcam;
- calls the detection function;
- manages the distraction timer;
- triggers the warning message;
- triggers the simulated safety action;
- displays the final annotated video.

### requirements.txt

This file contains the Python libraries required to run the scenario.

## Installation

From the project root folder, install the dependencies with:

```bash
pip install -r scenario_1_driver_distraction/requirements.txt
```

## Execution

From the project root folder, run:

```bash
python scenario_1_driver_distraction/main.py
```

## How to Test

1. Run the program.
2. Look normally at the webcam.
3. The system should display:

```text
DRIVER ATTENTIVE
```

4. Look down as if using a phone.
5. The system should display:

```text
WARNING: EYES ON ROAD
```

6. Keep looking down for more than 3 seconds.
7. The system should display:

```text
ACTION: DISENGAGING CRUISE CONTROL
SLOWING DOWN
```

8. Press `q` to quit the program.

## Expected Output

The program opens a webcam window and displays the driver's status in real time.

When the driver is attentive, the status is shown in green.

When the driver is distracted, the warning message is shown in red.

After 3 seconds of continuous distraction, the system simulates disengaging cruise control and slowing down the vehicle.

## Important Note

This project is an academic simulation. It does not control a real vehicle. The speed reduction and cruise control disengagement are simulated visually for demonstration purposes.

## Team 01 Contribution

This scenario was developed by Team 01 as part of the collaborative Autonomous Vehicle Perception project.

The work is divided as follows:

- `detector.py`: driver attention detection logic;
- `config.py`: system parameters and thresholds;
- `main.py`: real-time execution and warning escalation;
- `README.md`: documentation and usage instructions.