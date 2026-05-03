import threading
import queue
import time
import os
import platform


class VoiceEngine:
    """
    Single shared queue across all instances — no overlap, no parallel speech.
    Uses gTTS + playsound on Windows (most reliable for Arabic).
    Falls back to pyttsx3, then print.
    """
    _shared_queue  = queue.Queue(maxsize=5)
    _shared_lock   = threading.Lock()
    _last_said: dict = {}
    _worker_active = False
    _engine_cache  = None

    def __init__(self, lang="ar", repeat_gap=5.0):
        self.lang       = lang
        self.repeat_gap = repeat_gap
        with VoiceEngine._shared_lock:
            if VoiceEngine._engine_cache is None:
                VoiceEngine._engine_cache = self._init_engine()
            if not VoiceEngine._worker_active:
                VoiceEngine._worker_active = True
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()

    def _init_engine(self):
        # 1. gTTS — best Arabic quality, works on Windows without extra setup
        try:
            from gtts import gTTS
            import playsound as _ps
            return "gtts"
        except Exception:
            pass
        # 2. pyttsx3
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 145)
            voices = eng.getProperty("voices")
            for v in voices:
                name = (v.name or "").lower()
                vid  = (v.id  or "").lower()
                if "arabic" in name or "ar-" in vid or "\\ar\\" in vid:
                    eng.setProperty("voice", v.id)
                    break
            return ("pyttsx3", eng)
        except Exception:
            pass
        return "print"

    # ------------------------------------------------------------------
    def speak(self, text: str):
        if not text or not text.strip():
            return
        now = time.time()
        with VoiceEngine._shared_lock:
            if now - VoiceEngine._last_said.get(text, 0) < self.repeat_gap:
                return
            VoiceEngine._last_said[text] = now
        try:
            VoiceEngine._shared_queue.put_nowait(text)
        except queue.Full:
            pass

    def clear_queue(self):
        while not VoiceEngine._shared_queue.empty():
            try:
                VoiceEngine._shared_queue.get_nowait()
            except queue.Empty:
                break
        with VoiceEngine._shared_lock:
            VoiceEngine._last_said.clear()

    def stop(self):
        pass

    # ------------------------------------------------------------------
    def _worker(self):
        while True:
            try:
                text = VoiceEngine._shared_queue.get(timeout=0.5)
                self._say(text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceWorker] {e}")

    def _say(self, text: str):
        engine = VoiceEngine._engine_cache
        try:
            if engine == "gtts":
                self._say_gtts(text)
            elif isinstance(engine, tuple) and engine[0] == "pyttsx3":
                _, eng = engine
                eng.say(text)
                eng.runAndWait()
            else:
                print(f"[TTS] {text}")
        except Exception as e:
            print(f"[VoiceEngine._say] {e}")

    def _say_gtts(self, text: str):
        from gtts import gTTS
        import tempfile, subprocess

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            tts = gTTS(text=text, lang=self.lang, slow=False)
            tts.save(tmp)

            # Windows — use playsound
            if platform.system() == "Windows":
                try:
                    import playsound
                    playsound.playsound(tmp, block=True)
                    return
                except Exception:
                    pass
                # fallback: Windows Media Player via cmd
                try:
                    os.startfile(tmp)
                    time.sleep(3)
                    return
                except Exception:
                    pass

            # Linux/Mac — ffplay
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp],
                check=False
            )
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass