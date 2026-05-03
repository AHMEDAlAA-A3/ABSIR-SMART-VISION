from pydantic import BaseModel
from typing  import Optional, List, Any


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    name_en:    str
    name_ar:    str
    confidence: float
    bbox:       Optional[BBox] = None


class DangerInfo(BaseModel):
    name_en:    str
    name_ar:    str
    level:      str
    confidence: float
    size_ratio: float
    bbox:       BBox


class ABSIRResponse(BaseModel):
    """Unified response for ALL endpoints — REST and WebSocket."""
    status:     str                        # success | error
    mode:       str                        # object | currency | text | color
    input_type: str                        # upload | capture | stream
    message:    Optional[str]   = None     # Arabic voice message
    danger:     Optional[str]   = None     # low | medium | high | null
    detections: List[Detection] = []
    extra:      Optional[Any]   = None     # text/color specific data
    image_b64:  Optional[str]   = None     # annotated or corrected frame