# src/app.py
#
# Server FastAPI untuk Web Application Demo Inferensi AttentiveSkel-3D.
# Menggunakan Server-Sent Events (SSE) untuk menampilkan progress real-time
# di tiap tahapan pipeline (Ekstraksi → Preprocessing → Inferensi → Rendering).

import os
import json
import shutil
import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.inference_core_v2 import run_inference_pipeline

app = FastAPI(title="AttentiveSkel-3D Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
MODELS_DIR    = PROJECT_ROOT / "bobot_model"
OUTPUT_DIR    = PROJECT_ROOT / "data_sementara"
TEMPLATES_DIR = PROJECT_ROOT / "web_app" / "templates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = TEMPLATES_DIR / "index_v2.html"
    if not index_path.exists():
        return "File index_v2.html tidak ditemukan di web_app/templates/index_v2.html"
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/models")
async def get_models():
    if not MODELS_DIR.exists():
        return {"models": []}
    models = [f.name for f in MODELS_DIR.glob("*.pth")]
    return {"models": sorted(models)}


@app.post("/api/predict/stream")
async def predict_stream(
    video: UploadFile = File(...),
    model_name: str   = Form(...)
):
    """
    Endpoint streaming (Server-Sent Events).
    Menjalankan inferensi di background thread dan mengalirkan progress
    tiap tahap ke browser secara real-time.
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        async def err():
            yield f"data: {json.dumps({'done': True, 'success': False, 'error': f'Model {model_name} tidak ditemukan.'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    # Simpan video mentah
    temp_video_path = OUTPUT_DIR / video.filename
    with open(temp_video_path, "wb") as buf:
        shutil.copyfileobj(video.file, buf)

    # Queue untuk komunikasi thread → async generator
    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_cb(step: int, total: int, message: str, detail: str = ""):
        """Dipanggil dari thread inferensi, menaruh event ke queue."""
        event = {"step": step, "total": total, "message": message, "detail": detail}
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_pipeline():
        """Dijalankan di thread terpisah agar tidak memblokir event loop."""
        try:
            result = run_inference_pipeline(
                video_path=str(temp_video_path),
                model_path=str(model_path),
                output_dir=str(OUTPUT_DIR),
                progress_callback=progress_cb
            )
            out_filename = Path(result["video_path"]).name
            done_event = {
                "done"    : True,
                "success" : True,
                "video_url": f"/videos/{out_filename}",
                "n_benar" : result["n_benar"],
                "n_salah" : result["n_salah"],
            }
        except Exception as exc:
            done_event = {"done": True, "success": False, "error": str(exc)}
        loop.call_soon_threadsafe(queue.put_nowait, done_event)

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("done"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Legacy endpoint (kept for backward compatibility) ──────────────────────
@app.post("/api/predict")
async def predict(video: UploadFile = File(...), model_name: str = Form(...)):
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return {"error": f"Model {model_name} tidak ditemukan."}
    temp_video_path = OUTPUT_DIR / video.filename
    with open(temp_video_path, "wb") as buf:
        shutil.copyfileobj(video.file, buf)
    try:
        result = run_inference_pipeline(
            video_path=str(temp_video_path),
            model_path=str(model_path),
            output_dir=str(OUTPUT_DIR)
        )
        if isinstance(result, dict):
            out_path = result["video_path"]
            n_benar  = int(result.get("n_benar", 0))
            n_salah  = int(result.get("n_salah", 0))
        else:
            out_path = result; n_benar = n_salah = 0
        return {"success": True, "video_url": f"/videos/{Path(out_path).name}",
                "n_benar": n_benar, "n_salah": n_salah}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
