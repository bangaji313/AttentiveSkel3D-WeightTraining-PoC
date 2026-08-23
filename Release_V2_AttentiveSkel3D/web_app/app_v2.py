# web_app/app_v2.py  (updated — Phase 4)
#
# Server FastAPI untuk Web Application Demo Inferensi + Explanation Dashboard.
# Endpoint baru:
#   /api/explain/stream  — full explanation pipeline (SSE)
#   /api/compare/stream  — compare semua model (SSE)

import os
import json
import shutil
import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_APP_DIR  = Path(__file__).resolve().parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WEB_APP_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_APP_DIR))

from inference_core_v2 import run_inference_pipeline
from explainer_v2 import (
    run_all_models_compare,
    BIOMECHANICAL_REFERENCE,
    LANDMARK_NAMES,
)

app = FastAPI(title="AttentiveSkel-3D Explanation Dashboard V2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
MODELS_DIR    = PROJECT_ROOT / "bobot_model"
OUTPUT_DIR    = PROJECT_ROOT / "data_sementara"
TEMPLATES_DIR = PROJECT_ROOT / "web_app" / "templates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/videos",  StaticFiles(directory=str(OUTPUT_DIR)), name="videos")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = TEMPLATES_DIR / "index_v2.html"
    if not index_path.exists():
        return "File index_v2.html tidak ditemukan."
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/models")
async def get_models():
    if not MODELS_DIR.exists():
        return {"models": []}
    models = [f.name for f in MODELS_DIR.glob("*.pth")]
    return {"models": sorted(models)}


@app.get("/api/exercises")
async def get_exercises():
    return {"exercises": list(BIOMECHANICAL_REFERENCE.keys())}


@app.get("/api/landmark_names")
async def get_landmark_names():
    return {"names": LANDMARK_NAMES}


@app.get("/api/download_json")
async def download_json(stem: str):
    json_path = OUTPUT_DIR / f"{stem}_explanation.json"
    if not json_path.exists():
        return {"error": "File not found"}
    return FileResponse(
        path=str(json_path),
        media_type="application/json",
        filename=f"{stem}_explanation.json",
    )


# ===========================================================================
# /api/explain/stream — Full explanation pipeline (SSE) (Phase 4)
# ===========================================================================

@app.post("/api/explain/stream")
async def explain_stream(
    video: UploadFile = File(...),
    model_name: str   = Form(...),
    exercise: str     = Form("Squat"),
):
    """Streaming endpoint untuk full explanation pipeline."""
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        async def err():
            yield f"data: {json.dumps({'done': True, 'success': False, 'error': f'Model {model_name} tidak ditemukan.'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    temp_video_path = OUTPUT_DIR / video.filename
    with open(temp_video_path, "wb") as buf:
        shutil.copyfileobj(video.file, buf)

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_cb(step, total, message, detail=""):
        event = {"step": step, "total": total, "message": message, "detail": detail}
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_pipeline():
        try:
            result = run_inference_pipeline(
                video_path=str(temp_video_path),
                model_path=str(model_path),
                output_dir=str(OUTPUT_DIR),
                exercise=exercise,
                progress_callback=progress_cb,
            )
            out_filename = Path(result["video_path"]).name
            stem         = Path(temp_video_path).stem
            done_event = {
                "done":         True,
                "success":      True,
                "video_url":    f"/videos/{out_filename}",
                "json_stem":    stem,
                "n_benar":      result["n_benar"],
                "n_salah":      result["n_salah"],
                "explanation":  result["explanation"],
            }
        except Exception as exc:
            import traceback
            done_event = {"done": True, "success": False, "error": str(exc),
                          "traceback": traceback.format_exc()}
        loop.call_soon_threadsafe(queue.put_nowait, done_event)

    threading.Thread(target=run_pipeline, daemon=True).start()

    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("done"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===========================================================================
# /api/compare/stream — Compare all models (Phase 6) (SSE)
# ===========================================================================

@app.post("/api/compare/stream")
async def compare_stream(
    video: UploadFile = File(...),
    exercise: str     = Form("Squat"),
):
    """
    Jalankan tensor yang sama melalui semua 5 skenario model.
    Tidak mengubah output model masing-masing.
    """
    temp_video_path = OUTPUT_DIR / video.filename
    with open(temp_video_path, "wb") as buf:
        shutil.copyfileobj(video.file, buf)

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_compare():
        try:
            import torch
            import numpy as np
            from src.data.extract_pose import PoseExtractor
            from src.data.preprocess import DataPreprocessor

            # Report step 1
            loop.call_soon_threadsafe(queue.put_nowait, {
                "step": 1, "total": 3, "message": "Mengekstrak pose dari video...", "detail": ""
            })

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            stem   = temp_video_path.stem

            raw_npy  = OUTPUT_DIR / f"{stem}_cmp_raw.npy"
            t64_npy  = OUTPUT_DIR / f"{stem}_cmp_64.npy"

            extractor = PoseExtractor(model_complexity=2)
            extractor.extract_video(video_path=str(temp_video_path), output_npy_path=str(raw_npy))

            preprocessor = DataPreprocessor(target_frames=64)
            tensor_data  = preprocessor.process(npy_file_path=str(raw_npy), output_npy_path=str(t64_npy))
            input_tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0)

            loop.call_soon_threadsafe(queue.put_nowait, {
                "step": 2, "total": 3, "message": "Menjalankan semua 5 model...", "detail": ""
            })

            results = run_all_models_compare(MODELS_DIR, input_tensor, exercise, device)

            loop.call_soon_threadsafe(queue.put_nowait, {
                "done": True, "success": True, "results": results, "exercise": exercise
            })
        except Exception as exc:
            import traceback
            loop.call_soon_threadsafe(queue.put_nowait, {
                "done": True, "success": False, "error": str(exc),
                "traceback": traceback.format_exc()
            })

    threading.Thread(target=run_compare, daemon=True).start()

    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("done"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===========================================================================
# Legacy endpoint — backward compatible (Phase 3 original)
# ===========================================================================

@app.post("/api/predict/stream")
async def predict_stream(
    video: UploadFile = File(...),
    model_name: str   = Form(...),
    exercise: str     = Form("Squat"),
):
    """Legacy SSE endpoint — diteruskan ke /api/explain/stream."""
    return await explain_stream(video=video, model_name=model_name, exercise=exercise)


@app.post("/api/predict")
async def predict(video: UploadFile = File(...), model_name: str = Form(...),
                  exercise: str = Form("Squat")):
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return {"error": f"Model {model_name} tidak ditemukan."}
    temp_video_path = OUTPUT_DIR / video.filename
    with open(temp_video_path, "wb") as buf:
        shutil.copyfileobj(video.file, buf)
    try:
        result   = run_inference_pipeline(
            video_path=str(temp_video_path), model_path=str(model_path),
            output_dir=str(OUTPUT_DIR), exercise=exercise,
        )
        out_path = result["video_path"]
        return {
            "success":   True,
            "video_url": f"/videos/{Path(out_path).name}",
            "n_benar":   result["n_benar"],
            "n_salah":   result["n_salah"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
