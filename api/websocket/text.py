import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from api.services.processing_service import load_frame_from_b64

_sys = None


def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys


async def text_stream(websocket: WebSocket):
    """
    WS /ws/text/stream
    Continuous OCR on incoming frames.
    Useful for live reading (signs, labels, documents).
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"status": "ping"}))
                continue

            try:
                payload = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"status": "error", "message": "Invalid JSON"}))
                continue

            b64 = payload.get("image_b64", "")
            if not b64:
                await websocket.send_text(json.dumps({"status": "error", "message": "Missing image_b64"}))
                continue

            try:
                frame = load_frame_from_b64(b64)
            except Exception:
                await websocket.send_text(json.dumps({"status": "error", "message": "Bad frame"}))
                continue

            result = await loop.run_in_executor(
                None, sys().text_reader.read_image, frame
            )

            if result:
                resp = {
                    "status":     "success",
                    "mode":       "text",
                    "input_type": "stream",
                    "message":    result.get("message"),
                    "danger":     None,
                    "detections": [],
                    "extra":      {"text": result.get("text")},
                    "image_b64":  None,
                }
            else:
                resp = {
                    "status": "success", "mode": "text", "input_type": "stream",
                    "message": None, "danger": None, "detections": [],
                    "extra": {"text": None}, "image_b64": None,
                }

            await websocket.send_text(json.dumps(resp, ensure_ascii=False))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass