"""Runtime configuration, overridable via environment variables."""

import os
from pathlib import Path

# Where uploaded videos and intermediate/output audio files live.
WORK_DIR = Path(os.environ.get("UVR_WORK_DIR", "/tmp/uvr-jobs"))

# Where UVR / audio-separator model weights are cached.
MODEL_DIR = Path(os.environ.get("UVR_MODEL_DIR", "/tmp/audio-separator-models"))

# Default UVR vocal isolation model (MDX-Net architecture, CPU-friendly).
SEPARATION_MODEL = os.environ.get("UVR_SEPARATION_MODEL", "Kim_Vocal_2.onnx")

# Path to the standalone DeepFilterNet CLI binary (statically built, no
# Python/torch dependency — see Dockerfile). Its default embedded model is
# DeepFilterNet2.
DEEPFILTER_BIN = os.environ.get("UVR_DEEPFILTER_BIN", "deep-filter")

# Max upload size, in megabytes.
MAX_UPLOAD_MB = int(os.environ.get("UVR_MAX_UPLOAD_MB", "500"))

# Max source duration we'll process, in seconds (keeps CPU jobs bounded on Railway).
MAX_DURATION_SECONDS = int(os.environ.get("UVR_MAX_DURATION_SECONDS", "1800"))

# How long finished job artifacts stick around before the janitor deletes them.
JOB_TTL_SECONDS = int(os.environ.get("UVR_JOB_TTL_SECONDS", str(60 * 60)))

# How many jobs can run their heavy processing concurrently.
MAX_CONCURRENT_JOBS = int(os.environ.get("UVR_MAX_CONCURRENT_JOBS", "1"))

WORK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
