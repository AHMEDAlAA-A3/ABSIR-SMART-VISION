from ultralytics import YOLO
from utils.voice import VoiceEngine
from utils.drawing import draw_corner_box, draw_label_box
import cv2
import time


class CurrencyDetector:
    AR_CURRENCY = {
        "100Egp": "مية جنيه",
        "200Egp": "ميتين جنيه",
        "50Egp":  "خمسين جنيه",
        "20Egp":  "عشرين جنيه",
        "10Egp":  "عشرة جنيه",
        "5Egp":   "خمسة جنيه",
    }
    VALUES = {"100Egp": 100, "200Egp": 200, "50Egp": 50,
              "20Egp": 20, "10Egp": 10, "5Egp": 5}

    def __init__(self, model_path: str, conf: float = 0.5):
        self.model                   = YOLO(model_path)
        self.conf                    = conf
        self.voice                   = VoiceEngine()
        self.last_spoken_currencies: set = set()
        self.last_speak_time         = 0.0
        self.speak_interval          = 5.0
        self.font                    = cv2.FONT_HERSHEY_SIMPLEX

    def detect_currency(self, image_path_or_frame):
        frame   = cv2.imread(image_path_or_frame) if isinstance(image_path_or_frame, str) else image_path_or_frame
        results = self.model(frame, conf=self.conf)
        detected, total, raw = [], 0, []

        for r in results:
            for box in r.boxes:
                cls     = int(box.cls[0])
                name_en = self.model.names[cls]
                name_ar = self.AR_CURRENCY.get(name_en, name_en)
                val     = self.VALUES.get(name_en, 0)
                detected.append(name_ar)
                total += val
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw.append({
                    "name_en": name_en, "name_ar": name_ar,
                    "value": val, "confidence": round(float(box.conf[0]), 2),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

        if detected:
            unique = list(set(detected))
            msg = f"دي ورقة {unique[0]}" if len(unique) == 1 else \
                  "في " + " و ".join(unique) + (f". المجموع {total} جنيه" if total else "")
            self.voice.speak(msg)
            return {"detected": unique, "total": total, "message": msg, "detections": raw}
        return None

    def detect_frame(self, frame):
        results          = self.model(frame, conf=self.conf, verbose=False)
        detected_set     = set()
        total            = 0
        annotated_frame  = frame.copy()
        raw              = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls     = int(box.cls[0])
                conf    = float(box.conf[0])
                name_en = self.model.names[cls]
                name_ar = self.AR_CURRENCY.get(name_en, name_en)
                val     = self.VALUES.get(name_en, 0)
                detected_set.add(name_ar)
                total += val
                raw.append({
                    "name_en": name_en, "name_ar": name_ar,
                    "value": val, "confidence": round(conf, 2),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })
                annotated_frame = draw_corner_box(
                    annotated_frame, x1, y1, x2, y2, color=(0, 220, 0), thickness=3
                )
                draw_label_box(annotated_frame, name_en, name_ar, x1, y1, x2,
                               box_color=(0, 180, 0))

        now = time.time()
        if detected_set:
            time_passed = (now - self.last_speak_time) >= self.speak_interval
            changed     = detected_set != self.last_spoken_currencies
            if time_passed or changed:
                unique = list(detected_set)
                msg = f"دي عملة {unique[0]}" if len(unique) == 1 else \
                      "في " + " و ".join(unique) + (f". المجموع {total} جنيه" if total and len(unique) > 1 else "")
                self.voice.speak(msg)
                self.last_spoken_currencies = detected_set.copy()
                self.last_speak_time        = now
            return annotated_frame, raw

        return None, []