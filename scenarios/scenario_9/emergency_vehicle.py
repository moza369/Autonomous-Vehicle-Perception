"""Emergency Vehicle Detection Module.

Detects approaching emergency vehicles by analyzing flashing red/blue
light patterns, optionally confirmed by a siren heard on the microphone,
then simulates the yield manoeuvre: pull over to the right shoulder and stop.
"""

import time

import cv2
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


# Only bright, saturated red/blue pixels are kept: dimmer thresholds pick up
# light reflections inside the cabin and coloured objects in the street.
BLUE_LOW, BLUE_HIGH = (100, 150, 160), (140, 255, 255)
RED_LOW_1, RED_HIGH_1 = (0, 150, 160), (10, 255, 255)
RED_LOW_2, RED_HIGH_2 = (170, 150, 160), (180, 255, 255)
MIN_PIXELS = 50
MIN_BOX_SIZE = 15
ALTERNATION_THRESHOLD = 2
HISTORY_FRAMES = 30

AUDIO_RATE = 22050
AUDIO_BLOCK = 2048
SIREN_MIN_RMS = 0.015
SIREN_BAND_HZ = (500, 2000)
SIREN_BAND_RATIO = 0.5
SIREN_HOLD_S = 1.0

INITIAL_SPEED_KMH = 50.0
SPEED_STEP_KMH = 0.6
LATERAL_STEP = 0.02

ACTION_TEXT = "ACTION: EMERGENCY VEHICLE DETECTED - PULLING OVER TO RIGHT SHOULDER"


class EmergencyVehicleDetector:
    """Detects emergency vehicles via flashing lights and siren."""

    def __init__(self, audio_enabled=True):
        self._stream = None
        self._audio_ok = False
        self._base_volume = 0.01
        self._last_siren_time = 0.0
        if audio_enabled:
            self._start_audio()
        self.reset()

    def reset(self):
        self._historique = []
        self._vehicule_signale = False
        self._siren_at_detection = False
        self._speed = INITIAL_SPEED_KMH
        self._lateral = 0.0
        self._base_volume = 0.01
        self._last_siren_time = 0.0

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._audio_ok = False

    def __del__(self):
        self.close()

    def _start_audio(self):
        if sd is None:
            return
        try:
            self._stream = sd.InputStream(channels=1, samplerate=AUDIO_RATE,
                                          blocksize=AUDIO_BLOCK,
                                          callback=self._audio_callback)
            self._stream.start()
            self._audio_ok = True
        except Exception:
            self._stream = None
            self._audio_ok = False

    def _audio_callback(self, indata, frames, time_info, status):
        x = indata[:, 0].astype(np.float64)
        rms = float(np.sqrt(np.mean(x ** 2)))

        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / AUDIO_RATE)
        total = spectrum[1:].sum()
        band = spectrum[(freqs >= SIREN_BAND_HZ[0]) & (freqs <= SIREN_BAND_HZ[1])].sum()
        band_ratio = band / total if total > 0 else 0.0

        loud = rms > max(SIREN_MIN_RMS, self._base_volume * 3)
        if loud and band_ratio > SIREN_BAND_RATIO:
            self._last_siren_time = time.monotonic()
        elif not loud:
            self._base_volume = self._base_volume * 0.98 + rms * 0.02

    def _siren_detected(self):
        return (self._audio_ok and
                time.monotonic() - self._last_siren_time < SIREN_HOLD_S)

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
        blue_mask = cv2.inRange(hsv, np.array(BLUE_LOW), np.array(BLUE_HIGH))
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(RED_LOW_1), np.array(RED_HIGH_1)),
            cv2.inRange(hsv, np.array(RED_LOW_2), np.array(RED_HIGH_2)))

        for mask, color in ((blue_mask, (255, 0, 0)), (red_mask, (0, 0, 255))):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > MIN_BOX_SIZE and ch > MIN_BOX_SIZE:
                    cv2.rectangle(output, (x, y), (x + cw, y + ch), color, 2)

        red_count = cv2.countNonZero(red_mask)
        blue_count = cv2.countNonZero(blue_mask)
        if red_count < MIN_PIXELS and blue_count < MIN_PIXELS:
            state = 0
        elif red_count > blue_count:
            state = 1
        else:
            state = -1

        self._historique.append(state)
        self._historique = self._historique[-HISTORY_FRAMES:]

        # Frames with no light (between two flashes) are skipped, so a
        # red -> dark -> blue sequence still counts as one alternation.
        alternances = 0
        derniere = 0
        for d in self._historique:
            if d == 0:
                continue
            if derniere != 0 and d != derniere:
                alternances += 1
            derniere = d
        clignote = alternances >= ALTERNATION_THRESHOLD

        sirene = self._siren_detected()

        if clignote and not self._vehicule_signale:
            self._vehicule_signale = True
            self._siren_at_detection = sirene

        if self._vehicule_signale:
            self._lateral = min(1.0, self._lateral + LATERAL_STEP)
            self._speed = max(0.0, self._speed - SPEED_STEP_KMH)
            vehicle_state = "STOPPED" if self._speed == 0.0 else "PULLING_OVER"
        else:
            vehicle_state = "CRUISING"

        cv2.rectangle(output, (0, 0), (w, 70), (0, 0, 0), -1)
        if self._vehicule_signale:
            cv2.putText(output, ACTION_TEXT, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 2, cv2.LINE_AA)
            if vehicle_state == "STOPPED":
                status = "VEHICLE STOPPED ON RIGHT SHOULDER"
            else:
                status = "YIELDING: MOVING TO RIGHT SHOULDER"
            cv2.putText(output, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)

            arrow_y = h - 50
            start_x = w // 2 - 60
            end_x = start_x + 20 + int(self._lateral * 150)
            cv2.arrowedLine(output, (start_x, arrow_y), (end_x, arrow_y),
                            (0, 255, 255), 3, tipLength=0.3)
        else:
            cv2.putText(output, "MONITORING FOR EMERGENCY VEHICLES", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
            info = f"Alternations: {alternances}  Flashing: {clignote}"
            cv2.putText(output, info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1)

        if not self._audio_ok:
            siren_text = "SIREN: N/A (no microphone)"
        else:
            siren_text = "SIREN: DETECTED" if sirene else "SIREN: --"
        cv2.putText(output, siren_text, (w - 230, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255) if sirene else (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(output, f"SPEED: {self._speed:.0f} km/h", (w - 170, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if vehicle_state == "STOPPED" else (255, 255, 255),
                    2, cv2.LINE_AA)

        if not self._vehicule_signale:
            detection = "NONE"
        elif self._siren_at_detection or sirene:
            detection = "FLASHING LIGHTS + SIREN"
        else:
            detection = "FLASHING LIGHTS"

        telemetry = {
            "emergency_detected": self._vehicule_signale,
            "lights_flashing": clignote,
            "alternations": alternances,
            "siren_detected": sirene,
            "audio_available": self._audio_ok,
            "detection": detection,
            "vehicle_state": vehicle_state,
            "speed_kmh": round(self._speed, 1),
            "action": ACTION_TEXT if self._vehicule_signale else "MONITORING",
        }
        return output, telemetry
