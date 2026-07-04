"""FastAPI app: upload a video/audio file, isolate + clean up the
speech in it, download the result. See README.md for the full pipeline
description."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from app import config, pipeline
from app.jobs import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvr.main")

app = FastAPI(title="Voice Isolation Studio")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

MAX_UPLOAD_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
SAFE_SUFFIX = re.compile(r"[^A-Za-z0-9._-]+")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


async def _save_upload(file: UploadFile, dest: Path) -> int:
    written = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(CHUNK_SIZE):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File exceeds the {config.MAX_UPLOAD_MB}MB limit.")
            out.write(chunk)
    return written


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    noise_reduction: float = Form(55.0),
    low_cut_hz: float = Form(90.0),
    high_cut_hz: float = Form(7500.0),
    vocal_boost: float = Form(100.0),
    compression: float = Form(42.0),
    gain_db: float = Form(0.0),
    gate_threshold: float = Form(-60.0),
    use_ai_denoise: bool = Form(True),
    use_transcription: bool = Form(True),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    suffix = SAFE_SUFFIX.sub("_", Path(file.filename).suffix or ".bin")[:10]
    upload_path = config.WORK_DIR / f"upload-{uuid.uuid4().hex[:12]}{suffix}"

    try:
        written = await _save_upload(file, upload_path)
    except HTTPException:
        upload_path.unlink(missing_ok=True)
        raise
    except Exception:
        upload_path.unlink(missing_ok=True)
        logger.exception("upload failed")
        raise HTTPException(500, "Upload failed.")

    if written == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty.")

    tweak = pipeline.TweakParams(
        noise_reduction=noise_reduction,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        vocal_boost=vocal_boost,
        compression=compression,
        gain_db=gain_db,
        gate_threshold=gate_threshold,
        use_ai_denoise=use_ai_denoise,
        use_transcription=use_transcription,
    ).clamped()

    job = manager.create_job(upload_path, file.filename, tweak)
    return JSONResponse(job.to_public_dict(), status_code=202)


@app.post("/api/preview")
async def preview(
    file: UploadFile = File(...),
    noise_reduction: float = Form(55.0),
    low_cut_hz: float = Form(90.0),
    high_cut_hz: float = Form(7500.0),
    vocal_boost: float = Form(100.0),
    compression: float = Form(42.0),
    gain_db: float = Form(0.0),
    gate_threshold: float = Form(-60.0),
    use_ai_denoise: bool = Form(True),
) -> FileResponse:
    """Render just the first ~20s of the source through the real
    enhance+amplify chain so the user can audition slider settings quickly,
    without waiting for (or paying the cost of) processing the whole file."""
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    preview_dir = config.WORK_DIR / f"preview-{uuid.uuid4().hex[:12]}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    suffix = SAFE_SUFFIX.sub("_", Path(file.filename).suffix or ".bin")[:10]
    upload_path = preview_dir / f"src{suffix}"

    try:
        written = await _save_upload(file, upload_path)
    except HTTPException:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(preview_dir, ignore_errors=True)
        logger.exception("preview upload failed")
        raise HTTPException(500, "Upload failed.")

    if written == 0:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise HTTPException(400, "Uploaded file is empty.")

    tweak = pipeline.TweakParams(
        noise_reduction=noise_reduction,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        vocal_boost=vocal_boost,
        compression=compression,
        gain_db=gain_db,
        gate_threshold=gate_threshold,
        use_ai_denoise=use_ai_denoise,
        use_transcription=False,
    ).clamped()

    try:
        mp3_path = await asyncio.to_thread(pipeline.render_preview, upload_path, preview_dir, tweak, 20.0)
    except pipeline.PipelineError as exc:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise HTTPException(422, str(exc))
    except Exception:
        shutil.rmtree(preview_dir, ignore_errors=True)
        logger.exception("preview render failed")
        raise HTTPException(500, "Preview render failed.")

    return FileResponse(
        mp3_path,
        media_type="audio/mpeg",
        filename="preview.mp3",
        background=BackgroundTask(shutil.rmtree, preview_dir, ignore_errors=True),
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (it may have expired).")
    return job.to_public_dict()


@app.get("/api/jobs/{job_id}/download/{name}")
def download(job_id: str, name: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (it may have expired).")
    path = job.result_files.get(name)
    if path is None or not path.exists():
        raise HTTPException(404, "That file isn't available.")
    media_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".txt": "text/plain",
        ".srt": "application/x-subrip",
    }.get(path.suffix, "application/octet-stream")
    download_name = f"{Path(job.original_filename).stem}-{name}"
    return FileResponse(path, media_type=media_type, filename=download_name)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
