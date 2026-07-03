FROM python:3.11-slim

# ffmpeg/ffprobe for extraction + mastering; libsndfile for audio I/O;
# ca-certificates so the DeepFilterNet binary fetch below can verify TLS;
# build-essential because audio-separator's `diffq` dependency compiles a
# C extension (bitpack.c) at install time and python:3.11-slim ships no
# compiler by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      ca-certificates \
      build-essential \
      libopenblas-dev \
      libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVR_MODEL_DIR=/srv/models \
    UVR_WORK_DIR=/tmp/uvr-jobs \
    UVR_DEEPFILTER_BIN=/usr/local/bin/deep-filter

# DeepFilterNet's standalone Rust CLI: statically built, model weights
# embedded, no torch/torchaudio version dance required for denoising.
ARG DEEPFILTER_VERSION=0.5.6
ADD https://github.com/Rikorose/DeepFilterNet/releases/download/v${DEEPFILTER_VERSION}/deep-filter-${DEEPFILTER_VERSION}-x86_64-unknown-linux-musl \
    /usr/local/bin/deep-filter
RUN chmod +x /usr/local/bin/deep-filter

# Install a CPU-only torch/torchvision build first so the subsequent
# audio-separator install (which pins torch>=2.3 but not a variant)
# doesn't pull several GB of unused CUDA runtime packages.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY scripts ./scripts

# Bake the default UVR separation model into the image so the first
# upload doesn't pay for a cold model download.
RUN python scripts/prefetch_models.py

EXPOSE 8000

# Railway injects $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
