from ultralytics import YOLO
from utils.drawing import draw_corner_box, draw_label_box
from utils.voice import VoiceEngine
import cv2
import time


class ObjectDetector:
    AR_OBJECTS = {
        "person": "شخص", "car": "سيارة", "bicycle": "عجلة",
        "motorcycle": "موتوسيكل", "bus": "اوتوبيس", "truck": "شاحنة",
        "chair": "كرسي", "couch": "كنبة", "bed": "سرير",
        "dining table": "طاولة", "laptop": "لابتوب", "mouse": "ماوس",
        "keyboard": "كيبورد", "cell phone": "موبايل", "bottle": "زجاجة",
        "cup": "كوب", "fork": "شوكة", "knife": "سكينة",
        "spoon": "معلقة", "bowl": "طبق", "banana": "موزة",
        "apple": "تفاحة", "orange": "برتقانة", "book": "كتاب",
        "scissors": "مقص", "backpack": "شنطة", "handbag": "حقيبة",
        "tie": "كرافتة", "umbrella": "شمسية", "remote": "ريموت",
        "tv": "تلفزيون", "monitor": "شاشة",
    }

    def __init__(self, model_path: str, conf: float = 0.4):
        self.model              = YOLO(model_path)
        self.conf               = conf
        self.voice              = VoiceEngine()
        self.last_spoken_objects: set = set()
        self.last_speak_time    = 0.0
        self.speak_interval     = 5.0
        self.font               = cv2.FONT_HERSHEY_SIMPLEX
        self._last_results      = []

    def detect_frame(self, frame):
        results         = self.model(frame, conf=self.conf, verbose=False)
        self._last_results = results
        detected_objects: set = set()
        annotated_frame = frame.copy()
        raw_detections  = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls     = int(box.cls[0])
                conf    = float(box.conf[0])
                name_en = self.model.names[cls]
                name_ar = self.AR_OBJECTS.get(name_en, name_en)
                detected_objects.add(name_ar)
                raw_detections.append({
                    "name_en": name_en, "name_ar": name_ar,
                    "confidence": round(conf, 2),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

                annotated_frame = draw_corner_box(
                    annotated_frame, x1, y1, x2, y2, color=(255, 80, 80), thickness=2
                )
                draw_label_box(annotated_frame, name_en, name_ar, x1, y1, x2,
                               box_color=(255, 80, 80))

        now = time.time()
        if detected_objects:
            time_passed     = (now - self.last_speak_time) >= self.speak_interval
            objects_changed = detected_objects != self.last_spoken_objects
            if time_passed or objects_changed:
                items = list(detected_objects)[:3]
                msg   = f"شايف {items[0]}" if len(items) == 1 else "شايف " + " و ".join(items)
                self.voice.speak(msg)
                self.last_spoken_objects = detected_objects.copy()
                self.last_speak_time     = now
            return annotated_frame, raw_detections

        return None, []

    def get_last_results(self):
        return self._last_results, self.model.names