import cv2
from detectors.object_detector   import ObjectDetector
from detectors.currency_detector  import CurrencyDetector
from detectors.color_recognizer   import ColorRecognizer
from detectors.text_reader        import TextReader
from utils.danger_alert           import DangerAlert
from config.settings import (
    OBJECTS_MODEL_PATH, CURRENCY_MODEL_PATH, DANGER_COOLDOWN
)


class ABSIRSystem:
    def __init__(self):
        print("Loading ABSIR System...")
        self.object_detector   = ObjectDetector(OBJECTS_MODEL_PATH)
        self.currency_detector = CurrencyDetector(CURRENCY_MODEL_PATH)
        self.color_recognizer  = ColorRecognizer()
        self.text_reader       = TextReader()
        self.danger_alert      = DangerAlert(cooldown=DANGER_COOLDOWN)
        print("ABSIR Ready!")

    # ------------------------------------------------------------------
    # IMAGE MODE  →  returns clean JSON-safe dict
    # ------------------------------------------------------------------
    def process_image(self, image_path: str, mode: str = "auto") -> dict:
        frame = cv2.imread(image_path)
        if frame is None:
            return {"status": "error", "message": "Cannot read image"}

        if mode == "currency":
            result = self.currency_detector.detect_currency(frame)
            return {"status": "success", "mode": "currency",
                    "result": result or {}, "danger": None}

        if mode == "object":
            _, detections = self.object_detector.detect_frame(frame)
            danger = self.danger_alert.process(
                self.object_detector._last_results,
                self.object_detector.model.names,
                frame.shape
            )
            return {"status": "success", "mode": "object",
                    "detections": detections, "danger": danger}

        if mode == "text":
            result = self.text_reader.read_image(frame)
            return {"status": "success", "mode": "text",
                    "result": result or {}, "danger": None}

        if mode == "color":
            result = self.color_recognizer.detect_dominant_color(image_path)
            return {"status": "success", "mode": "color",
                    "result": result, "danger": None}

        # AUTO
        result = self.currency_detector.detect_currency(frame)
        if result:
            return {"status": "success", "mode": "currency",
                    "result": result, "danger": None}

        _, detections = self.object_detector.detect_frame(frame)
        if detections:
            danger = self.danger_alert.process(
                self.object_detector._last_results,
                self.object_detector.model.names,
                frame.shape
            )
            return {"status": "success", "mode": "object",
                    "detections": detections, "danger": danger}

        result = self.text_reader.read_image(frame)
        if result:
            return {"status": "success", "mode": "text",
                    "result": result, "danger": None}

        result = self.color_recognizer.detect_dominant_color(image_path)
        return {"status": "success", "mode": "color",
                "result": result, "danger": None}

    # ------------------------------------------------------------------
    # FRAME MODE  →  always returns (annotated_frame, detections, danger)
    # ------------------------------------------------------------------
    def process_frame(self, frame, mode: str = "auto"):
        """
        Returns: (annotated_frame, detections: list, danger: dict | None)
        annotated_frame is always a valid numpy array.
        """
        if frame is None:
            return frame, [], None

        if mode == "currency":
            ann, raw = self.currency_detector.detect_frame(frame)
            return (ann if ann is not None else frame), raw, None

        if mode == "object":
            ann, raw = self.object_detector.detect_frame(frame)
            danger   = self.danger_alert.process(
                self.object_detector._last_results,
                self.object_detector.model.names,
                frame.shape
            )
            return (ann if ann is not None else frame), raw, danger

        # AUTO  (currency → object; text is image-only)
        ann, raw = self.currency_detector.detect_frame(frame)
        if ann is not None:
            return ann, raw, None

        ann, raw = self.object_detector.detect_frame(frame)
        danger   = self.danger_alert.process(
            self.object_detector._last_results,
            self.object_detector.model.names,
            frame.shape
        )
        return (ann if ann is not None else frame), raw, danger