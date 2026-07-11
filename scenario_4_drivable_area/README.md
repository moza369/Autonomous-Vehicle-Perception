# Scenario 4 - Drivable Area / Free Space Detection

This project detects the drivable area of the road using a YOLOv8 segmentation model.

The system identifies lane boundaries and recommends steering corrections to keep the vehicle centered.

Example actions:

- ACTION: KEEP LANE
- ACTION: STEERING LEFT
- ACTION: STEERING RIGHT

## Project Description

The system processes road images and performs segmentation to identify the drivable area.

Based on the detected lane position, it recommends one of the following actions:

- KEEP LANE
- STEERING LEFT - CORRECTING LANE DEPARTURE
- STEERING RIGHT - CORRECTING LANE DEPARTURE

## Project Structure

```text
scenario_4_drivable_area/
├── datasets/
├── models/
├── results/
├── src/
├── requirements.txt
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Model

The trained model is **not included** because it exceeds GitHub's recommended file size.

See:

```text
models/README.md
```

for download instructions.

## Dataset

The full BDD100K dataset is also **not included**.

See:

```text
datasets/README.md
```

for setup instructions.

A sample image is included for quick testing.

## Run

```bash
python src/main.py
```

## Technologies

- Python
- OpenCV
- NumPy
- ONNX Runtime
- YOLOv8 Segmentation

## Case Study

On a curved highway, the vehicle begins drifting toward the right shoulder.

The system detects the drivable area and recommends a steering correction to keep the vehicle centered.

## Authors

JIMI & CHAHID