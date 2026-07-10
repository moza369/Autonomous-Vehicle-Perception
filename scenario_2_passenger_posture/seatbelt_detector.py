class SeatbeltDetector:
    def __init__(self):
        # Par defaut la ceinture est attachee
        self.status = "ON"

    def toggle(self):
        # Permet de basculer l'etat de la ceinture
        if self.status == "ON":
            self.status = "OFF"
        else:
            self.status = "ON"

    def get_status(self):
        return self.status