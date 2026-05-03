import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routes.object   import router as object_router
from api.routes.currency import router as currency_router
from api.routes.text     import router as text_router
from api.routes.color    import router as color_router

from api.websocket.object   import object_stream
from api.websocket.text     import text_stream
from api.websocket.color   import color_stream
from api.websocket.currency import currency_stream

from api.services.camera_service import camera
from config.settings import API_HOST, API_PORT, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    camera.open(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)

    from api.routes.object import sys as obj_sys
    obj_sys()   # pre-load YOLO models

    yield

    # Shutdown
    camera.release()


app = FastAPI(
    title       = "ABSIR API",
    description = "AI Vision Assistance System for the Blind",
    version     = "3.0",
    lifespan    = lifespan,
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── REST routers ──────────────────────────────────────────────────────
app.include_router(object_router)
app.include_router(currency_router)
app.include_router(text_router)
app.include_router(color_router)

# ── WebSocket endpoints ───────────────────────────────────────────────
@app.websocket("/ws/object/stream")
async def ws_object(ws: WebSocket):
    await object_stream(ws)

@app.websocket("/ws/text/stream")
async def ws_text(ws: WebSocket):
    await text_stream(ws)

@app.websocket("/ws/currency/stream")
async def ws_currency(ws: WebSocket):
    await currency_stream(ws)

@app.websocket("/ws/color/stream")
async def ws_color(ws: WebSocket, type: str = "protanopia"):
    await color_stream(ws, cb_type=type)

# ── Health ────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"name": "ABSIR API", "version": "3.0", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "ok", "camera": camera.is_open}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True, workers=1)