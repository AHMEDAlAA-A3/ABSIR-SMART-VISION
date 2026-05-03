from fastapi import APIRouter, File, UploadFile, Form, Query
from api.services.camera_service      import camera
from api.services.processing_service  import load_frame_from_upload
from detectors.color_blind            import process as cb_process, TYPES

router = APIRouter(prefix="/api/color", tags=["Color Blind Assistance"])
_sys   = None


def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys


def _build(frame, input_type: str, cb_type: str) -> dict:
    result = cb_process(frame, cb_type)
    return {
        "status":     "success",
        "mode":       "color",
        "input_type": input_type,
        "message":    f"تم تصحيح الصورة لنوع عمى الألوان: {result['struggle']}",
        "danger":     None,
        "detections": [],
        "extra": {
            "type":             result["type"],
            "struggle":         result["struggle"],
            "original_colors":  result["original_colors"],
            "corrected_colors": result["corrected_colors"],
            "simulated_b64":    result["simulated_b64"],
        },
        "image_b64": result["corrected_b64"],
    }


@router.post("/upload")
async def color_upload(
    file:    UploadFile = File(...),
    type:    str        = Query("protanopia", enum=TYPES),
):
    """
    type: protanopia | deuteranopia | tritanopia
    Returns corrected image + color analysis.
    """
    return _build(load_frame_from_upload(file), "upload", type)


@router.post("/capture")
async def color_capture(type: str = Query("protanopia", enum=TYPES)):
    frame = camera.capture()
    if frame is None:
        return {"status": "error", "message": "Camera not available"}
    return _build(frame, "capture", type)