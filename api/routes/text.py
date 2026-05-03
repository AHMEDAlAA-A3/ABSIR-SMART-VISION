from fastapi import APIRouter, File, UploadFile, Form
from api.services.camera_service      import camera
from api.services.processing_service  import load_frame_from_upload

router = APIRouter(prefix="/api/text", tags=["Text Recognition"])
_sys   = None


def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys


def _build(frame, input_type: str) -> dict:
    result = sys().text_reader.read_image(frame)
    if not result:
        return {
            "status": "success", "mode": "text", "input_type": input_type,
            "message": "لم يتم اكتشاف نص", "danger": None,
            "detections": [], "extra": {"text": None}, "image_b64": None,
        }
    return {
        "status":     "success",
        "mode":       "text",
        "input_type": input_type,
        "message":    result.get("message"),
        "danger":     None,
        "detections": [],
        "extra":      {"text": result.get("text"), "raw_text": result.get("raw_text")},
        "image_b64":  None,
    }


@router.post("/upload")
async def text_upload(file: UploadFile = File(...)):
    return _build(load_frame_from_upload(file), "upload")


@router.post("/capture")
async def text_capture():
    frame = camera.capture()
    if frame is None:
        return {"status": "error", "message": "Camera not available"}
    return _build(frame, "capture")