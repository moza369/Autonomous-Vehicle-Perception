"""Emergency Vehicle Detection Module.

Detects approaching emergency vehicles by analyzing flashing blue/red
light patterns.
"""

import cv2
import numpy as np

class EmergencyVehicleDetector:
    """Detects emergency vehicles via flashing lights."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._historique = []
        self._vehicule_signale = False
        self._fps = 30

    def process_frame(self, frame):
        """Process a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        telemetry : dict
        """
        frame = cv2.resize(frame, (640, 360))
        h, w = frame.shape[:2]
        output = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv,
                                np.array([100, 150, 50]),
                                np.array([140, 255, 255]))

        red_mask1 = cv2.inRange(hsv,
                                np.array([0, 150, 50]),
                                np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv,
                                np.array([170, 150, 50]),
                                np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        for mask, color, label in [(blue_mask, (255, 0, 0), "BLUE"),
                                    (red_mask, (0, 0, 255), "RED")]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > 15 and ch > 15:
                    cv2.rectangle(output, (x, y), (x + cw, y + ch), color, 2)

        red_count = cv2.countNonZero(red_mask)
        blue_count = cv2.countNonZero(blue_mask)

        if red_count < 50 and blue_count < 50:
            state = 0
        elif red_count > blue_count:
            state = 1   # Red dominant
        else:
            state = -1  # Blue dominant

        fenetre = max(10, self._fps)
        self._historique.append(state)
        if len(self._historique) > fenetre:
            self._historique = self._historique[-int(fenetre):]

        alternances = 0
        for i in range(1, len(self._historique)):
            if (self._historique[i] != 0 and self._historique[i - 1] != 0 and
                    self._historique[i] != self._historique[i - 1]):
                alternances += 1

        clignote = alternances >= 2

        if clignote:
            self._vehicule_signale = True

        cv2.rectangle(output, (0, 0), (w, 70), (0, 0, 0), -1)

        if self._vehicule_signale:
            cv2.putText(output,
                        "ACTION: EMERGENCY VEHICLE DETECTED - PULLING OVER TO RIGHT SHOULDER",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(output, "EMERGENCY VEHICLE ALERT",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)

            arrow_cx = w // 2
            arrow_cy = h - 50
            cv2.arrowedLine(output,
                            (arrow_cx - 60, arrow_cy),
                            (arrow_cx + 60, arrow_cy),
                            (0, 255, 255), 3, tipLength=0.3)
        else:
            cv2.putText(output, "MONITORING FOR EMERGENCY VEHICLES",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            info = f"Alternations: {alternances}  Flashing: {clignote}"
            cv2.putText(output, info, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        telemetry = {
            "emergency_detected": self._vehicule_signale,
            "lights_flashing": clignote,
            "alternations": alternances,
            "action": ("ACTION: EMERGENCY VEHICLE DETECTED - "
                       "PULLING OVER TO RIGHT SHOULDER"
                       if self._vehicule_signale
                       else "MONITORING"),
        }
        return output, telemetry
