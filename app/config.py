"""Runtime configuration, overridable via environment variables."""

import os
from pathlib import Path

# Where uploaded videos and intermediate/output audio files live.
WORK_DIR = Path(os.environ.get("UVR_WORK_DIR", "/tmp/uvr-jobs"))

# Max upload size, in megabytes.
MAX_UPLOAD_MB = int(os.environ.get("UVR_MAX_UPLOAD_MB", "500"))

# Max source duration we'll process, in seconds (keeps CPU jobs bounded on Railway).
MAX_DURATION_SECONDS = int(os.environ.get("UVR_MAX_DURATION_SECONDS", "3600"))

# How long finished job artifacts stick around before the janitor deletes them.
JOB_TTL_SECONDS = int(os.environ.get("UVR_JOB_TTL_SECONDS", str(60 * 60)))

# How many jobs can run their heavy processing concurrently.
MAX_CONCURRENT_JOBS = int(os.environ.get("UVR_MAX_CONCURRENT_JOBS", "2"))

WORK_DIR.mkdir(parents=True, exist_ok=True)
