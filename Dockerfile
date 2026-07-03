FROM python:3.11-slim

# ffmpeg for all audio I/O + amplification; curl/ca-certificates to fetch the
# DeepFilterNet binary; git/cmake/build-essential to build whisper.cpp.
# NOTE: there is deliberately NO python ML stack (torch/onnxruntime/numpy) --
# that was the sole cause of the "import numpy failed" runtime crashes. Both AI
# engines here are self-contained native binaries with zero python deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
      curl \
      git \
      cmake \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVR_WORK_DIR=/tmp/uvr-jobs \
    UVR_DEEPFILTER_BIN=/usr/local/bin/deep-filter \
    UVR_WHISPER_BIN=/usr/local/bin/whisper-cli \
    UVR_WHISPER_MODEL=/srv/models/ggml-base.en.bin

# DeepFilterNet3 speech-enhancement CLI: statically built Rust binary with the
# model embedded (no python/torch/numpy). This is the primary noise-removal
# engine -- it pulls faint speech out of non-stationary noise (traffic, crowd)
# far better than ffmpeg's generic denoisers.
ARG DEEPFILTER_VERSION=0.5.6
ADD https://github.com/Rikorose/DeepFilterNet/releases/download/v${DEEPFILTER_VERSION}/deep-filter-${DEEPFILTER_VERSION}-x86_64-unknown-linux-musl \
    /usr/local/bin/deep-filter
RUN chmod +x /usr/local/bin/deep-filter

# whisper.cpp for transcription/captions (self-contained C++ binary, no numpy).
# Best-effort: the whole step is wrapped so a build/download hiccup can never
# fail the image -- transcription just gets skipped at runtime if absent.
ARG WHISPER_MODEL=base.en
RUN bash -c 'set -eux; \
      git clone --depth 1 https://github.com/ggml-org/whisper.cpp /tmp/whisper.cpp; \
      cmake -S /tmp/whisper.cpp -B /tmp/whisper.cpp/build \
            -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON; \
      cmake --build /tmp/whisper.cpp/build -j --config Release --target whisper-cli; \
      cp /tmp/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli; \
      mkdir -p /srv/models; \
      curl -fSL -o /srv/models/ggml-'"${WHISPER_MODEL}"'.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-'"${WHISPER_MODEL}"'.bin; \
      rm -rf /tmp/whisper.cpp; \
      /usr/local/bin/whisper-cli --help >/dev/null 2>&1 && echo "whisper OK"; \
    ' || echo "WHISPER SETUP FAILED — transcription disabled (audio pipeline unaffected)"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

EXPOSE 8000

# Railway injects $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
