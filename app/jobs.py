"""In-memory async job manager.

Jobs are CPU-heavy (ML inference + ffmpeg) so they run on a small
thread pool and report progress the frontend polls for. State lives in
a process-local dict, which is fine for a single Railway instance and
avoids needing Redis/Postgres for an MVP; see README for the scaling
note if this ever needs multiple instances.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app import config, pipeline

logger = logging.getLogger("uvr.jobs")

STAGES = [
    "queued",
    "extracting audio",
    "isolating voices",
    "removing background noise",
    "boosting quiet speech",
    "done",
    "error",
]


@dataclass
class Job:
    id: str
    original_filename: str
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    stage_progress: float = 0.0
    error: Optional[str] = None
    work_dir: Optional[Path] = None
    result_files: dict[str, Path] = field(default_factory=dict)

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.original_filename,
            "status": self.status,
            "progress": round(self.stage_progress, 3),
            "error": self.error,
            "downloads": sorted(self.result_files.keys()) if self.status == "done" else [],
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_JOBS)
        self._janitor = threading.Thread(target=self._janitor_loop, daemon=True)
        self._janitor.start()

    def create_job(self, upload_path: Path, original_filename: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        work_dir = config.WORK_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        job = Job(id=job_id, original_filename=original_filename, work_dir=work_dir)
        with self._lock:
            self._jobs[job_id] = job

        self._executor.submit(self._run, job, upload_path)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _set_stage(self, job: Job, status: str, progress: float = 0.0) -> None:
        job.status = status
        job.stage_progress = progress
        logger.info("job %s -> %s (%.0f%%)", job.id, status, progress * 100)

    def _run(self, job: Job, upload_path: Path) -> None:
        assert job.work_dir is not None
        work_dir = job.work_dir
        try:
            self._set_stage(job, "extracting audio", 0.0)
            source_info = pipeline.probe_source(upload_path)
            if source_info.duration_seconds > config.MAX_DURATION_SECONDS:
                raise pipeline.PipelineError(
                    f"Source is {source_info.duration_seconds / 60:.1f} min long; "
                    f"the limit is {config.MAX_DURATION_SECONDS / 60:.0f} min."
                )

            original_wav = work_dir / "original.wav"
            pipeline.extract_audio(upload_path, original_wav)
            job.result_files["original.wav"] = original_wav

            self._set_stage(job, "isolating voices", 0.0)
            vocals_path = pipeline.separate_vocals(
                original_wav,
                work_dir / "separated",
                progress_cb=lambda _stage, pct: self._set_stage(job, "isolating voices", pct),
            )

            self._set_stage(job, "removing background noise", 0.0)
            denoised_path = pipeline.denoise_speech(
                vocals_path,
                work_dir,
                progress_cb=lambda _stage, pct: self._set_stage(
                    job, "removing background noise", pct
                ),
            )

            self._set_stage(job, "boosting quiet speech", 0.0)
            final_wav = work_dir / "voice_clean.wav"
            final_mp3 = work_dir / "voice_clean.mp3"
            pipeline.master_speech(
                denoised_path,
                final_wav,
                final_mp3,
                progress_cb=lambda _stage, pct: self._set_stage(
                    job, "boosting quiet speech", pct
                ),
            )
            job.result_files["voice_clean.wav"] = final_wav
            job.result_files["voice_clean.mp3"] = final_mp3

            self._set_stage(job, "done", 1.0)
        except pipeline.PipelineError as exc:
            job.error = str(exc)
            self._set_stage(job, "error", 0.0)
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors to the UI too
            logger.exception("job %s crashed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"[:500]
            self._set_stage(job, "error", 0.0)
        finally:
            if upload_path.exists():
                upload_path.unlink(missing_ok=True)

    def _janitor_loop(self) -> None:
        while True:
            time.sleep(300)
            cutoff = time.time() - config.JOB_TTL_SECONDS
            with self._lock:
                stale = [j for j in self._jobs.values() if j.created_at < cutoff]
            for job in stale:
                if job.work_dir and job.work_dir.exists():
                    shutil.rmtree(job.work_dir, ignore_errors=True)
                with self._lock:
                    self._jobs.pop(job.id, None)
                logger.info("janitor: purged job %s", job.id)


manager = JobManager()
