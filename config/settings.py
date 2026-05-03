import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

# موديلات
CURRENCY_MODEL_PATH = "D:\Study\FinalGraduationProject\ABSIR PROJECT\detectors\currency_detector.py"
OBJECTS_MODEL_PATH = "D:\Study\FinalGraduationProject\DataSet\Data\yolov8l.pt"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Detection
CURRENCY_CONFIDENCE = 0.5
OBJECT_CONFIDENCE = 0.4

# Voice
VOICE_LANGUAGE = "ar"
VOICE_ENABLED = True

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Uploads
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DANGER_COOLDOWN = 4