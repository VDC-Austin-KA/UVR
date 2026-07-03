FROM python:3.11-slim

# ffmpeg/ffprobe do ALL of the audio work (extraction, denoising, voice
# amplification). No machine-learning stack is installed on purpose: the
# previous torch/onnxruntime/numpy dependency chain was the sole source of
# runtime "import numpy failed" crashes, and ffmpeg's built-in denoisers and
# dynamics deliver the result with none of that fragility.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVR_WORK_DIR=/tmp/uvr-jobs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

EXPOSE 8000

# Railway injects $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
