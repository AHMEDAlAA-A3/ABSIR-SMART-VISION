import pytesseract
import cv2
from utils.voice import VoiceEngine

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_OK = True
except Exception:
    ARABIC_OK = False


class TextReader:
    TESS_CMD_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def __init__(self):
        import os, platform
        if platform.system() == "Windows" and os.path.exists(self.TESS_CMD_WIN):
            pytesseract.pytesseract.tesseract_cmd = self.TESS_CMD_WIN

        self.voice        = VoiceEngine(lang="ar")
        self.last_spoken  = None

    def read_image(self, image_path_or_frame) -> dict | None:
        frame = cv2.imread(image_path_or_frame) \
            if isinstance(image_path_or_frame, str) else image_path_or_frame

        gray  = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 3)
        text  = pytesseract.image_to_string(gray, lang="ara+eng").strip()

        if len(text) < 3:
            return None

        display_text = text
        if ARABIC_OK:
            try:
                display_text = get_display(arabic_reshaper.reshape(text))
            except Exception:
                pass

        msg = f"النص المكتوب: {text}"
        if msg != self.last_spoken:
            self.voice.speak(msg)
            self.last_spoken = msg

        return {"text": display_text, "raw_text": text, "message": msg}