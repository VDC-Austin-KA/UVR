"""Runtime configuration, overridable via environment variables."""

import os
import shutil
from pathlib import Path

# Where uploaded videos and intermediate/output audio files live.
WORK_DIR = Path(os.environ.get("UVR_WORK_DIR", "/tmp/uvr-jobs"))

# DeepFilterNet standalone CLI (statically built Rust binary, DeepFilterNet3
# model embedded, NO python/torch/numpy dependency). This is the primary
# speech-enhancement engine. If the binary is missing the pipeline falls back
# to ffmpeg-only denoising.
DEEPFILTER_BIN = os.environ.get("UVR_DEEPFILTER_BIN", "/usr/local/bin/deep-filter")

# whisper.cpp CLI + ggml model for transcription (also numpy-free). Best-effort:
# if either is missing, transcription is skipped and the audio still returns.
WHISPER_BIN = os.environ.get("UVR_WHISPER_BIN", "/usr/local/bin/whisper-cli")
WHISPER_MODEL = os.environ.get("UVR_WHISPER_MODEL", "/srv/models/ggml-base.en.bin")

# Max upload size, in megabytes.
MAX_UPLOAD_MB = int(os.environ.get("UVR_MAX_UPLOAD_MB", "500"))

# Max source duration we'll process, in seconds (keeps CPU jobs bounded on Railway).
MAX_DURATION_SECONDS = int(os.environ.get("UVR_MAX_DURATION_SECONDS", "3600"))

# How long finished job artifacts stick around before the janitor deletes them.
JOB_TTL_SECONDS = int(os.environ.get("UVR_JOB_TTL_SECONDS", str(60 * 60)))

# How many jobs can run their heavy processing concurrently.
MAX_CONCURRENT_JOBS = int(os.environ.get("UVR_MAX_CONCURRENT_JOBS", "1"))


def deepfilter_available() -> bool:
    return bool(shutil.which(DEEPFILTER_BIN) or Path(DEEPFILTER_BIN).exists())


def whisper_available() -> bool:
    have_bin = bool(shutil.which(WHISPER_BIN) or Path(WHISPER_BIN).exists())
    return have_bin and Path(WHISPER_MODEL).exists()


WORK_DIR.mkdir(parents=True, exist_ok=True)
