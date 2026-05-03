"""
api/services/camera_service.py
Singleton camera — opened once, never duplicated.
Safe release on shutdown via lifespan.
"""
import cv2
import threading


class CameraService:
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._cap      = None
                obj._cam_lock = threading.Lock()
                cls._instance = obj
        return cls._instance

    def open(self, index: int = 0, width: int = 640, height: int = 480):
        with self._cam_lock:
            if self._cap and self._cap.isOpened():
                return True
            self._cap = cv2.VideoCapture(index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return self._cap.isOpened()

    def capture(self):
        """Grab one frame. Returns numpy array or None."""
        with self._cam_lock:
            if not self._cap or not self._cap.isOpened():
                return None
            ret, frame = self._cap.read()
            return frame if ret else None

    def release(self):
        with self._cam_lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
                self._cap = None

    @property
    def is_open(self) -> bool:
        return bool(self._cap and self._cap.isOpened())


camera = CameraService()