"""FastAPI app: upload a video/audio file, isolate + clean up the
speech in it, download the result. See README.md for the full pipeline
description."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
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


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    suffix = SAFE_SUFFIX.sub("_", Path(file.filename).suffix or ".bin")[:10]
    upload_path = config.WORK_DIR / f"upload-{uuid.uuid4().hex[:12]}{suffix}"

    written = 0
    try:
        with open(upload_path, "wb") as out:
            while chunk := await file.read(CHUNK_SIZE):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"File exceeds the {config.MAX_UPLOAD_MB}MB limit."
                    )
                out.write(chunk)
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

    job = manager.create_job(upload_path, file.filename)
    return JSONResponse(job.to_public_dict(), status_code=202)


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
