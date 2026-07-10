class SafetySystem:
    def __init__(self, current_speed=80):
        self.state = "NORMAL"
        self.airbag = "ENABLED"
        self.speed_limit = None
        self.current_speed = current_speed

    def update(self, seatbelt, posture_info):
        out_of_position = posture_info["out_of_position"]
        seatbelt_off = (seatbelt == "OFF")

        # Cas normal : tout est OK
        if not seatbelt_off and not out_of_position:
            self.state = "NORMAL"
            self.airbag = "ENABLED"
            self.speed_limit = None

        # Ceinture détachee mais posture correcte
        elif seatbelt_off and not out_of_position:
            self.state = "WARNING"
            self.airbag = "ENABLED"
            self.speed_limit = self.current_speed

        # Posture dangereuse mais ceinture attachée
        elif not seatbelt_off and out_of_position:
           self.state = "WARNING"
           self.airbag = "ENABLED"
           self.speed_limit = None

        # Ceinture détachee et mauvaise posture
        else:
            self.state = "DANGER"
            self.airbag = "DISABLED"
            self.speed_limit = self.current_speed

        return self.state

    def get_airbag_status(self):
        return self.airbag

    def get_speed_limit(self):
        return self.speed_limit
