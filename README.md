# Voice Isolation Studio

Upload a video (or audio file). This service:

1. **Extracts the audio at the highest practical quality** — lossless 48kHz/24-bit PCM, straight from the source via `ffmpeg`.
2. **Isolates the voices** from music, ambience, and background noise (traffic, wind, room tone, etc.) using a UVR (Ultimate Vocal Remover) MDX-Net model, via the [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) library — a Python wrapper around the same models developed in the [UVR project](https://github.com/Anjok07/ultimatevocalremovergui) (`Anjok07`), which this app's repo tracks and deploys.
3. **Denoises the isolated speech** with [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)'s standalone Rust CLI binary — a deep-learning speech enhancement model (weights embedded in the binary, no Python/torch dependency) that strips residual broadband/environmental noise while preserving the voice.
4. **Boosts quiet, barely-audible speech** with an `ffmpeg` mastering chain (`highpass` → `afftdn` → `speechnorm` → `loudnorm`) so faint dialogue becomes intelligible without clipping louder parts, then normalizes to a consistent, comfortable loudness (EBU R128, -16 LUFS).
5. Hands you back a **lossless WAV master** and a **compressed MP3** for quick mobile playback/download, plus the original extracted audio for comparison.

It's a plain, mobile-friendly web page — no app-store install needed. Open the
Railway URL on your iPhone in Safari and use it like any other site; tap
**Share → Add to Home Screen** if you want it to behave like an app icon.

## Why these particular tools

`lalal.ai` (the reference point named in the original request) is a paid,
closed-source SaaS API — there's no model to self-host from it. This app
instead builds the same *kind* of pipeline (extract → isolate voice →
denoise → normalize) entirely on open-source components so it can be
self-hosted on Railway with no third-party API keys or per-minute billing:
UVR's separation models for the isolation step, and DeepFilterNet for the
denoise step.

## Architecture

```
static/            mobile-first upload UI (vanilla HTML/CSS/JS)
app/main.py         FastAPI routes: upload, job status polling, downloads
app/jobs.py          in-memory job manager + background thread pool
app/pipeline.py       ffmpeg extraction, UVR separation, DeepFilterNet denoise, mastering
scripts/prefetch_models.py   downloads the UVR model at Docker build time
Dockerfile / railway.toml     container + Railway deploy config
```

Jobs run on a small thread pool (`UVR_MAX_CONCURRENT_JOBS`, default 1 — CPU
inference is the bottleneck) and report progress that the frontend polls.
State is in-memory and per-instance, which is intentional: this is a
single-service MVP with no database. If you outgrow one instance, swap the
`JobManager` for a queue (e.g. Redis + RQ) and move job files to object
storage — the `pipeline.py` functions are already decoupled from the web
layer so that's a contained change.

## Running locally

Requires `ffmpeg` on PATH (`apt install ffmpeg` / `brew install ffmpeg`), and
the [DeepFilterNet CLI binary](https://github.com/Rikorose/DeepFilterNet/releases)
(`deep-filter-<version>-<platform>`) downloaded and on PATH as `deep-filter`
(or point `UVR_DEEPFILTER_BIN` at wherever you put it).

```bash
# Install a CPU-only torch build first (skip this if you have a GPU and
# want audio-separator to use it — just `pip install torch torchvision` normally).
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/prefetch_models.py   # optional: warm the UVR model cache up front
uvicorn app.main:app --reload
```

Then open http://localhost:8000.

`scripts/smoke_test.py` exercises the whole web app (upload → job
progress → downloads) end to end with the two model-dependent stages
stubbed out; useful for verifying the FastAPI/job-manager layer without
needing the UVR/DeepFilterNet weights on hand:

```bash
python scripts/smoke_test.py /path/to/any/test/video.mp4
```

## Deploying to Railway

1. Push this repo to GitHub (already done if you're reading this from the
   repo) and create a new Railway project from it.
2. Railway auto-detects `railway.toml` and builds via the `Dockerfile` — no
   extra configuration needed. The model is baked into the image at build
   time (see `scripts/prefetch_models.py`), so cold starts don't pay for a
   model download.
3. Railway sets `$PORT` automatically; the container listens on it.
4. Once deployed, open the generated `*.up.railway.app` URL — on desktop or
   on your iPhone in Safari.

### Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `UVR_SEPARATION_MODEL` | `Kim_Vocal_2.onnx` | Which UVR MDX-Net model to use for voice isolation. See `audio-separator --list_models` for alternatives. |
| `UVR_MAX_UPLOAD_MB` | `500` | Reject uploads larger than this. |
| `UVR_MAX_DURATION_SECONDS` | `1800` | Reject sources longer than this (keeps CPU jobs bounded). |
| `UVR_MAX_CONCURRENT_JOBS` | `1` | Thread pool size for processing. Raise only if your Railway plan has the CPU/RAM to match. |
| `UVR_JOB_TTL_SECONDS` | `3600` | How long finished job files stick around before cleanup. |

## Limitations

- **CPU-only inference.** Railway services don't come with a GPU, so
  processing time scales with clip length — expect roughly real-time-ish
  throughput for the default model on a few vCPUs. There's a duration cap
  (`UVR_MAX_DURATION_SECONDS`) to keep jobs from running away.
- **Single instance, in-memory job state.** Don't scale this service
  horizontally without also moving job state out of process (see
  Architecture above) — a second instance won't see jobs created on the
  first.
- **Voice isolation, not diarization.** The pipeline isolates "voice" as a
  class of sound (vs. music/ambience/noise); it doesn't separate multiple
  overlapping speakers from each other.
