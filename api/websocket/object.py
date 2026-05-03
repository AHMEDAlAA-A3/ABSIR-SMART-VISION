import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from api.services.processing_service import load_frame_from_b64, frame_to_b64, clean_det, clean_danger

_sys = None


def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys


async def object_stream(websocket: WebSocket):
    """
    WS /ws/object/stream
    Client sends: { "image_b64": "...", "return_image": true }
    Server sends: unified ABSIRResponse JSON
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

            b64          = payload.get("image_b64", "")
            return_image = payload.get("return_image", True)

            if not b64:
                await websocket.send_text(json.dumps({"status": "error", "message": "Missing image_b64"}))
                continue

            try:
                frame = load_frame_from_b64(b64)
            except Exception:
                await websocket.send_text(json.dumps({"status": "error", "message": "Bad frame"}))
                continue

            ann, detections, danger = await loop.run_in_executor(
                None, sys().process_frame, frame, "object"
            )

            level, _ = clean_danger(danger)
            names    = [d["name_ar"] for d in (detections or [])[:3]]
            message  = None
            if danger:
                message = f"خلي بالك، أمامك {danger['name_ar']}"
            elif names:
                message = "شايف " + " و ".join(names)

            resp = {
                "status":     "success",
                "mode":       "object",
                "input_type": "stream",
                "message":    message,
                "danger":     level,
                "detections": [clean_det(d) for d in (detections or [])],
                "extra":      None,
                "image_b64":  frame_to_b64(ann, 65) if return_image else None,
            }
            await websocket.send_text(json.dumps(resp, ensure_ascii=False))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass