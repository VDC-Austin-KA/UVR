# Voice Isolation Studio

Upload a video or audio file with faint, hard-to-hear voices in a noisy
environment (traffic, wind, crowd, room noise). This service:

1. **Extracts the audio** at high fidelity (48kHz / 24-bit PCM), straight from
   the source via `ffmpeg`.
2. **Reduces the background noise** — bandpasses the signal to the voice range
   and runs `ffmpeg`'s adaptive FFT denoiser (`afftdn`, two passes) plus a
   non-local-means pass (`anlmdn`) to subtract steady traffic/wind/hiss/hum.
3. **Amplifies the quiet speech** so barely-audible voices become clearly
   hearable — `speechnorm` lifts quiet syllables, `dynaudnorm` evens out the
   level over time, `loudnorm` normalizes to a consistent loudness, and a
   limiter keeps the boosted result from clipping.
4. Hands you back a **lossless WAV** and a **compressed MP3**, plus the
   original extracted audio for comparison.

It's a plain, mobile-friendly web page — no app-store install needed. Open the
Railway URL on your iPhone in Safari and use it like any other site; tap
**Share → Add to Home Screen** if you want it to behave like an app icon.

## Why pure ffmpeg (no ML stack)

An earlier version used a Python machine-learning stack (torch + onnxruntime +
`audio-separator`/UVR) for model-based vocal separation. That stack repeatedly
failed **at runtime** on the host with an opaque `ImportError: import numpy
failed` coming out of onnxruntime — a failure that only reproduced inside the
deployment environment, not in local or build-time runs, which made it
effectively undebuggable.

This version does all of the work with `ffmpeg`, which is already installed and
has capable built-in denoisers and dynamics processors. The result: **no numpy,
no torch, no onnxruntime, no model downloads** — a small image that builds in
under a minute and can't hit that class of failure. It won't separate music
from vocals the way a dedicated source-separation model would, but for the
target use case — lifting faint speech out of ambient/broadband noise — it is
reliable and effective.

## Architecture

```
static/            mobile-first upload UI (vanilla HTML/CSS/JS)
app/main.py         FastAPI routes: upload, job status polling, downloads
app/jobs.py          in-memory job manager + background thread pool
app/pipeline.py       ffmpeg extraction, denoise, and voice amplification
Dockerfile / railway.toml     container + Railway deploy config
```

Jobs run on a small thread pool (`UVR_MAX_CONCURRENT_JOBS`) and report progress
the frontend polls for. State is in-memory and per-instance — this is a
single-service MVP with no database; the `pipeline.py` functions are decoupled
from the web layer, so moving to a queue + object storage later is contained.

## Running locally

Requires `ffmpeg` on PATH (`apt install ffmpeg` / `brew install ffmpeg`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000.

`scripts/smoke_test.py` runs the whole app end to end — it synthesizes a noisy
"faint speech" clip, uploads it, waits for the job, and downloads the results
(nothing is stubbed, because the pipeline has no external models):

```bash
python scripts/smoke_test.py                     # synthetic noisy test clip
python scripts/smoke_test.py /path/to/clip.mp4   # or your own file
```

## Deploying to Railway

1. Push this repo to GitHub and create a Railway project from it.
2. Railway auto-detects `railway.toml` and builds via the `Dockerfile` — no
   extra configuration needed.
3. Railway sets `$PORT` automatically; the container listens on it.
4. Open the generated `*.up.railway.app` URL — on desktop or on your iPhone.

### Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `UVR_MAX_UPLOAD_MB` | `500` | Reject uploads larger than this. |
| `UVR_MAX_DURATION_SECONDS` | `3600` | Reject sources longer than this. |
| `UVR_MAX_CONCURRENT_JOBS` | `2` | Thread pool size for processing. |
| `UVR_JOB_TTL_SECONDS` | `3600` | How long finished job files stick around before cleanup. |

## Roadmap

- **Transcription / captions.** Automatic speech-to-text of the cleaned audio,
  planned via a self-contained (no-numpy) engine such as `whisper.cpp`, kept
  best-effort so it can never break the core audio pipeline.

## Limitations

- **CPU-only processing.** ffmpeg is fast, but very long clips still take time;
  there's a duration cap (`UVR_MAX_DURATION_SECONDS`) to keep jobs bounded.
- **Single instance, in-memory job state.** Don't scale horizontally without
  moving job state out of process.
- **Noise reduction, not source separation.** This lifts speech out of
  ambient/broadband noise; it does not separate overlapping speakers or strip
  musical accompaniment the way a dedicated separation model would.
