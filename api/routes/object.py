from fastapi import APIRouter, File, UploadFile, Form
from api.services.camera_service    import camera
from api.services.processing_service import (
    load_frame_from_upload, frame_to_b64, clean_det, clean_danger
)

router = APIRouter(prefix="/api/object", tags=["Object Detection"])
_sys   = None


def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys


def _build(frame, input_type: str, return_image: bool) -> dict:
    ann, detections, danger = sys().process_frame(frame, mode="object")
    level, _ = clean_danger(danger)
    names    = [d["name_ar"] for d in (detections or [])[:3]]
    message  = None
    if danger:
        message = f"خلي بالك، أمامك {danger['name_ar']}"
    elif names:
        message = "شايف " + " و ".join(names)

    return {
        "status":     "success",
        "mode":       "object",
        "input_type": input_type,
        "message":    message,
        "danger":     level,
        "detections": [clean_det(d) for d in (detections or [])],
        "extra":      None,
        "image_b64":  frame_to_b64(ann) if return_image else None,
    }


# ── Upload ───────────────────────────────────────────────────────────
@router.post("/upload")
async def object_upload(
    file:         UploadFile = File(...),
    return_image: bool       = Form(False),
):
    frame = load_frame_from_upload(file)
    return _build(frame, "upload", return_image)


# ── Capture ──────────────────────────────────────────────────────────
@router.post("/capture")
async def object_capture(return_image: bool = Form(False)):
    frame = camera.capture()
    if frame is None:
        return {"status": "error", "message": "Camera not available"}
    return _build(frame, "capture", return_image)