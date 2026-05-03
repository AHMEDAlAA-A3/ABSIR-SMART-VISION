import cv2
import numpy as np
from utils.voice import VoiceEngine


class ColorRecognizer:
    def __init__(self):
        self.voice = VoiceEngine()

    def detect_dominant_color(self, image_path) -> dict:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (100, 100))
        avg = np.mean(img.reshape(-1, 3), axis=0)

        b, g, r = avg
        if r > g and r > b and r > 150:
            color_en, color_ar = "red",   "أحمر"
        elif g > r and g > b and g > 150:
            color_en, color_ar = "green", "أخضر"
        elif b > r and b > g and b > 150:
            color_en, color_ar = "blue",  "أزرق"
        else:
            color_en, color_ar = "unknown", "غير معروف"

        self.voice.speak(f"اللون الغالب هو {color_ar}")
        return {"color_en": color_en, "color_ar": color_ar}