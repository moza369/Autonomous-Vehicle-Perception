# Models

This directory contains the models used by Scenario 7.

## YOLOv8n

The system uses the YOLOv8 Nano model for real-time object detection.

The model is used to detect:

- Pedestrians
- Sports balls
- Other vulnerable road users supported by the COCO classes

The model weights (`yolov8n.pt`) are not required to be committed to the repository if they are downloaded automatically by Ultralytics.

When running the project, make sure `yolov8n.pt` is available in the expected model path.
