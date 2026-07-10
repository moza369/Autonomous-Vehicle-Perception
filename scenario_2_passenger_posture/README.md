# Scenario 2 – Occupant Posture & Dynamic Seatbelt Detection (Out-of-Position)

## Objective

This module implements **Scenario 2** of the **Autonomous Vehicle Perception** project.

The objective is to detect when the **front passenger** is in an unsafe situation while the vehicle is moving.

The system continuously monitors:

* the passenger posture using computer vision;
* the passenger seatbelt status.

If the passenger unbuckles the seatbelt while simultaneously adopting an unsafe posture (for example leaning toward the center console or entering the airbag deployment zone), the system immediately displays a warning and simulates the appropriate ADAS safety actions.

---

# Scenario Description

The vehicle is assumed to be travelling at **80 km/h**.

If the passenger:

* unbuckles the seatbelt;
* leans forward toward the airbag deployment zone;
* or leans excessively toward the center console;
* or places the feet on the dashboard,

the system displays:

```
WARNING:
PASSENGER UNBUCKLED & OUT OF POSITION
```

The system also displays:

```
ACTION:
PASSENGER AIRBAG DISABLED
SPEED RESTRICTED TO 80 KM/H
```

The warning remains active until the passenger returns to a safe position and fastens the seatbelt again.

---

# Used Technologies

## OpenCV

OpenCV is used to:

* open the webcam;
* capture video frames in real time;
* display the visual interface;
* draw warning messages;
* manage keyboard input.

---

## MediaPipe Pose

MediaPipe Pose is used to estimate the passenger body landmarks.

These landmarks allow the system to detect:

* forward leaning;
* sideways leaning;
* feet on the dashboard;
* overall passenger posture.

---

## NumPy

NumPy is used for numerical computations required by the posture analysis.

---

# Detection Method

The system combines two complementary detection modules.

## 1. Passenger Posture Detection

MediaPipe Pose extracts body landmarks from each video frame.

The passenger is considered **out of position** when one of the following conditions is detected:

* leaning forward into the passenger airbag deployment zone;
* leaning excessively toward the center console;
* placing the feet on the dashboard.

---

## 2. Seatbelt Status

The passenger seatbelt status is simulated using the **S** keyboard key.

This allows testing the complete ADAS decision logic.

**In a real autonomous vehicle, the seatbelt status would be provided by the seatbelt buckle sensor connected to the vehicle CAN Bus.**

---

# Decision Rule

The passenger is considered in a dangerous situation when:

* Seatbelt = OFF

AND

* Passenger posture = OUT OF POSITION

When both conditions are satisfied, the system displays:

```
WARNING:
PASSENGER UNBUCKLED & OUT OF POSITION
```

and activates the following simulated safety actions:

```
ACTION:
PASSENGER AIRBAG DISABLED
SPEED RESTRICTED TO 80 KM/H
```

An audio warning is also played continuously until the passenger returns to a safe condition.

---

# Project Structure

```
Projet_ADAS_Posture_Ceinture/
│
├── main.py
├── pose_detector.py
├── seatbelt_detector.py
├── state_machine.py
├── audio.py
├── requirements.txt
└── README.md
```

---

# File Description

## main.py

Runs the complete real-time ADAS simulation.

Responsibilities:

* opens the webcam;
* processes each frame;
* calls the posture detector;
* reads the seatbelt status;
* updates the ADAS state machine;
* displays warnings and safety actions.

---

## pose_detector.py

Contains the MediaPipe Pose detection logic.

Responsibilities:

* body landmark extraction;
* forward posture detection;
* sideways posture detection;
* feet-on-dashboard detection.

---

## seatbelt_detector.py

Simulates the passenger seatbelt status.

The seatbelt state can be toggled using the **S** key.

---

## state_machine.py

Implements the ADAS decision logic.

Possible states:

* NORMAL
* WARNING
* DANGER

It also controls:

* warning messages;
* airbag status;
* speed restriction.

---

## audio.py

Plays a continuous warning sound while the passenger remains in a dangerous situation.

---

## requirements.txt

Contains all required Python dependencies.

---

# Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# Execution

Run the application using:

```bash
python main.py
```

---

# How to Test

1. Start the application.
2. The passenger is initially seated normally with the seatbelt fastened.
3. Press **S** to simulate unbuckling the seatbelt.
4. Lean forward or sideways in front of the webcam.
5. The system should display:

```
WARNING:
PASSENGER UNBUCKLED & OUT OF POSITION
```

6. The system then simulates:

```
ACTION:
PASSENGER AIRBAG DISABLED
SPEED RESTRICTED TO 80 KM/H
```

7. Return to a normal posture and press **S** again to fasten the seatbelt.

8. The system returns to the NORMAL state.

Press **ESC** to exit.

---

# Expected Output

The application opens a webcam window and continuously monitors the passenger.

When the passenger is safe, the interface displays:

```
NORMAL
```

When an unsafe posture or an unbuckled seatbelt is detected, warning messages appear.

When both conditions occur simultaneously, the system simulates:

* passenger airbag deactivation;
* vehicle speed restriction;
* continuous audio warning.

---

# Important Note

This project is an academic simulation.

The passenger posture is detected automatically using **MediaPipe Pose**.

The seatbelt status is simulated through keyboard input because a standard webcam cannot determine whether the seatbelt buckle is physically locked.

In a production vehicle, this information would be obtained from the dedicated seatbelt sensor connected to the vehicle CAN Bus.

The objective of this project is to demonstrate the complete perception and decision-making pipeline of an ADAS system rather than control a real vehicle.

---
## Future Improvements

- Detect the real seatbelt status using dedicated vehicle sensors.
- Improve posture estimation under different lighting conditions.
- Integrate the system with real ADAS hardware.
  
# Team Contribution

This scenario was developed as part of the collaborative **Autonomous Vehicle Perception** project

The work is organized as follows:

* **pose_detector.py** – passenger posture detection;
* **seatbelt_detector.py** – seatbelt status simulation;
* **state_machine.py** – ADAS decision logic;
* **audio.py** – warning sound management;
* **main.py** – real-time execution;
* **README.md** – project documentation.
* **team 2**
