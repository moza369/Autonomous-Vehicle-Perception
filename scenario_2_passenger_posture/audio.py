import sys
import threading
import time

class Alarm:
    def __init__(self):
        self.enabled = True
        self.en_cours = False

    def play(self):
        if self.enabled and not self.en_cours:
            self.en_cours = True
            # Lancer le bip sonore dans un thread pour ne pas bloquer l'affichage
            threading.Thread(target=self._beep, daemon=True).start()

    def _beep(self):
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(1000, 300)
            else:
                # Cloche terminale (Mac/Linux)
                print("\a", end="", flush=True)
        except Exception:
            pass
        time.sleep(1.0)
        self.en_cours = False

    def stop(self):
        self.enabled = False