"""Manages scenario detector instances and their lifecycle."""
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ScenarioManager:
    """Lazily loads and manages scenario detector instances."""
    
    def __init__(self):
        self._detectors = {}
        self._errors = {}
    
    def get_detector(self, scenario_key):
        """Get or create a detector instance for the given scenario."""
        if scenario_key in self._errors:
            return None, self._errors[scenario_key]
        
        if scenario_key not in self._detectors:
            try:
                self._detectors[scenario_key] = self._create_detector(scenario_key)
            except Exception as e:
                error_msg = f"Failed to load {scenario_key}: {e}\n{traceback.format_exc()}"
                self._errors[scenario_key] = error_msg
                return None, error_msg
        
        return self._detectors[scenario_key], None
    
    def _create_detector(self, scenario_key):
        """Create a new detector instance."""
        from config.settings import MODELS_DIR
        
        if scenario_key == "driver_distraction":
            from scenarios.scenario_1.driver_distraction import DriverDistractionDetector
            return DriverDistractionDetector()
        
        elif scenario_key == "passenger_safety":
            from scenarios.scenario_2.passenger_safety import PassengerSafetyDetector
            return PassengerSafetyDetector()
        
        elif scenario_key == "traffic_sign_recognition":
            from scenarios.scenario_3.traffic_sign_recognition import TSRDetector
            model_path = os.path.join(MODELS_DIR, "tsr_best.pt")
            return TSRDetector(model_path=model_path)
        
        elif scenario_key == "drivable_area":
            from scenarios.scenario_4.drivable_area import DrivableAreaDetector
            model_path = os.path.join(MODELS_DIR, "yolov8n-seg.onnx")
            return DrivableAreaDetector(model_path=model_path)
        
        elif scenario_key == "road_surface":
            from scenarios.scenario_5.road_surface import RoadSurfaceDetector
            return RoadSurfaceDetector()
        
        elif scenario_key == "vru_detection":
            from scenarios.scenario_7.vru_detector import VRUDetector
            model_path = os.path.join(MODELS_DIR, "yolov8n.pt")
            return VRUDetector(model_path=model_path)
        
        elif scenario_key == "vehicle_intent":
            from scenarios.scenario_8.vehicle_intent import VehicleIntentDetector
            model_path = os.path.join(MODELS_DIR, "yolov8n.pt")
            return VehicleIntentDetector(model_path=model_path)
        
        elif scenario_key == "emergency_vehicle":
            from scenarios.scenario_9.emergency_vehicle import EmergencyVehicleDetector
            return EmergencyVehicleDetector()
        
        elif scenario_key == "construction_zone":
            from scenarios.scenario_6.construction_detector import ConstructionZoneDetector
            return ConstructionZoneDetector()
        
        else:
            raise ValueError(f"Unknown scenario: {scenario_key}")
    
    def reset_detector(self, scenario_key):
        """Reset a detector's state."""
        if scenario_key in self._detectors:
            detector = self._detectors[scenario_key]
            if hasattr(detector, 'reset'):
                detector.reset()
    
    def release_detector(self, scenario_key):
        """Release a detector to free memory."""
        if scenario_key in self._detectors:
            del self._detectors[scenario_key]
        if scenario_key in self._errors:
            del self._errors[scenario_key]
    
    def release_all(self):
        """Release all detectors."""
        self._detectors.clear()
        self._errors.clear()
