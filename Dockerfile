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

# Install CPU-only torch AND torchvision from the same index so they are an
# ABI-matched pair. torchvision is required by onnx2torch, which audio-separator
# imports when loading MDX models; if torchvision instead comes from the default
# PyPI index (built against a different torch) its C ops fail to register and
# model loading dies with "operator torchvision::nms does not exist". Installing
# from the CPU wheel index also avoids pulling several GB of unused CUDA
# packages. torch drags in numpy 2.x here; requirements.txt pins it back to a
# numpy<2 / onnxruntime pair that audio-separator supports.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Exercise the full model-loading import chain at build time so a broken build
# fails here instead of at runtime on the first job. This covers both failure
# modes seen in practice: onnxruntime's numpy C-API import ("import numpy
# failed") and onnx2torch's torchvision op registration ("operator
# torchvision::nms does not exist").
RUN python -c "import numpy, onnxruntime, torchvision, onnx2torch; from audio_separator.separator.architectures.mdx_separator import MDXSeparator; print('deps OK -> numpy', numpy.__version__, '/ onnxruntime', onnxruntime.__version__, '/ torchvision', torchvision.__version__)"

COPY app ./app
COPY static ./static
COPY scripts ./scripts

# Bake the default UVR separation model into the image so the first
# upload doesn't pay for a cold model download.
RUN python scripts/prefetch_models.py

EXPOSE 8000

# Railway injects $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
