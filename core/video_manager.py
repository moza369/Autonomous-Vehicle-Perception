"""Manages video, image, and camera input sources."""
import cv2
import os


class VideoManager:
    """Handles opening/reading/releasing video sources."""
    
    def __init__(self):
        self._cap = None
        self._source = None
        self._is_camera = False
        self._total_frames = 0
        self._current_frame = 0
        self._fps = 30.0
    
    def open_video(self, path):
        """Open a video file."""
        self.release()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self._source = path
        self._is_camera = False
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._current_frame = 0
        return True
    
    def open_camera(self, index=0):
        """Open a live camera."""
        self.release()
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self._source = index
        self._is_camera = True
        self._total_frames = 0
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._current_frame = 0
        return True
    
    def read_frame(self):
        """Read the next frame."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if ret:
            self._current_frame += 1
        return ret, frame
    
    def read_image(self, path):
        """Read a single image."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return img
    
    def release(self):
        """Release the current source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._source = None
        self._current_frame = 0
    
    @property
    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def is_camera(self):
        return self._is_camera
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def total_frames(self):
        return self._total_frames
    
    @property
    def current_frame(self):
        return self._current_frame
    
    @property
    def progress(self):
        if self._total_frames > 0:
            return self._current_frame / self._total_frames
        return 0.0
