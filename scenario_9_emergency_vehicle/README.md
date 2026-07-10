# Scenario 9 — Emergency Vehicle Detection

## Team 09

| Name | GitHub |
|------|--------|
| Abdelilah Touhami | @Abdelilah1-h1 |
| Mohamed Alaoui | @alaouimed97 |

---

## Description

This scenario detects emergency vehicles (ambulances, police cars, fire trucks) approaching from behind using a rear camera feed.

The system combines two detection methods:
- **Visual detection**: flashing red/blue lights (gyrophares) alternating pattern
- **Audio detection**: siren sound via microphone

When both signals are detected simultaneously, the system triggers an alert and instructs the autonomous vehicle to pull over to the right shoulder.

---

## Files

| File | Description |
|------|-------------|
| `scenario 9.py` | Main detection script |
| `ambulance.mp4` | Test video of an approaching ambulance |
| `requirements.txt` | Python dependencies |

---

## How It Works

1. Captures frames from webcam (or video file)
2. Converts each frame to HSV color space
3. Detects red and blue pixel regions (emergency light colors)
4. Tracks alternation between red and blue over a sliding window
5. Simultaneously monitors microphone volume for siren detection
6. If **flashing lights AND siren** are both detected → triggers alert

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

**With webcam (real-time):**
```bash
python "scenario 9.py"
```

**With video file:**

In `scenario 9.py`, replace line 7:
```python
# Change this:
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# To this:
video = cv2.VideoCapture("ambulance.mp4")
```

Press **Q** to quit.

---

## Output

When an emergency vehicle is detected:
- Red alert text displayed: `ACTION: EMERGENCY VEHICLE DETECTED - PULLING OVER TO RIGHT SHOULDER`
- Arrow animation showing the vehicle moving to the right

---

## Testing

Tested with:
- Live webcam feed with ambulance video played on another screen
- Direct video file playback (`ambulance.mp4`)
