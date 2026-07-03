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


def _warm_up_ml_stack() -> None:
    """Import the heavy ML stack once, in the main thread, at boot.

    Two reasons this runs at import time rather than only lazily inside a
    worker thread:
    1. It initializes numpy's C-API in the main thread, so the first job's
       worker-thread import can't hit any first-import-in-a-thread issue.
    2. If numpy/onnxruntime can't load in this environment, it logs the full,
       real traceback to the startup logs -- instead of the opaque
       "import numpy failed" that onnxruntime raises at job time. It is wrapped
       so a failure here never stops the app from starting (the health check
       must still pass); jobs will then surface the real reason too.
    """
    try:
        import numpy
        import onnxruntime
        from audio_separator.separator import Separator  # noqa: F401

        logger.info(
            "ML stack warm: numpy %s / onnxruntime %s",
            numpy.__version__,
            onnxruntime.__version__,
        )
    except Exception:
        logger.exception("ML stack failed to import at startup")


_warm_up_ml_stack()

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
    media_type = "audio/wav" if path.suffix == ".wav" else "audio/mpeg"
    download_name = f"{Path(job.original_filename).stem}-{name}"
    return FileResponse(path, media_type=media_type, filename=download_name)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
