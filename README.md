# Voice Isolation Studio

Upload a video or audio file with faint, hard-to-hear voices in a noisy
environment (traffic, wind, crowd, room noise). This service:

1. **Extracts the audio** at high fidelity (48kHz / 24-bit PCM), straight from
   the source via `ffmpeg`.
2. **Removes the background noise with DeepFilterNet3** — a deep-learning
   speech enhancer (self-contained Rust binary, model embedded, no
   python/torch/numpy) that predicts a suppression filter per frequency bin.
   It handles non-stationary noise (traffic, crowd, wind) far better than
   generic denoisers, and removing the noise *first* is what lets the next step
   boost the voice without boosting the noise. Falls back to `ffmpeg` denoisers
   if the binary is unavailable.
3. **Amplifies the quiet speech** so barely-audible voices become clearly
   hearable — `speechnorm` lifts quiet syllables, an `acompressor` +
   `dynaudnorm` even out the dynamics, `loudnorm` normalizes to a loud,
   consistent level, and a limiter keeps it from clipping.
4. **Transcribes the speech with Whisper** (`whisper.cpp` — a self-contained
   C++ binary, no numpy) into a readable transcript plus `.srt` captions.
   Whisper is remarkably good at reading speech that is barely audible to the
   ear. Best-effort: if unavailable, the cleaned audio still returns.
5. Hands you back a **lossless WAV** and a **compressed MP3**, the transcript
   and captions, plus the original extracted audio for comparison.

It's a plain, mobile-friendly web page — no app-store install needed. Open the
Railway URL on your iPhone in Safari and use it like any other site; tap
**Share → Add to Home Screen** if you want it to behave like an app icon.

## Tuning sliders

After picking a file, a tuning panel appears with sliders for every processing
knob before you commit to the full run:

| Slider | Effect |
|---|---|
| Noise reduction | Strength of the ffmpeg spectral denoise pass (on top of DeepFilterNet) |
| Low cut / High cut | Band-limits the signal to the voice range |
| Vocal boost | How hard quiet speech gets lifted |
| Compression | Evens out loud vs. quiet syllables |
| Overall volume | Final loudness target shift |
| Noise gate | Cuts residual hiss between words |
| AI noise removal | Toggles DeepFilterNet3 on/off |
| Transcribe speech | Toggles Whisper on/off (skip it for a faster run) |

Two ways to audition a setting before running the full file:
- **Live rough preview** — dragging a slider updates a Web Audio graph
  (EQ/gain/compression) on the locally loaded file *instantly*, no server
  round-trip. It's an approximation (no real spectral noise reduction happens
  in-browser) meant for fast, rough tuning by ear.
- **Render accurate preview** — sends the current slider values to the server,
  which runs the *real* chain (ffmpeg + DeepFilterNet if enabled) on just the
  first 20 seconds and hands back a short MP3. This is what the full run will
  actually sound like.

Once you're happy, **Process full file with these settings** runs the complete
pipeline with the chosen values. The plain **Clean up the audio** button on the
upload screen skips tuning and uses sensible defaults.

## Why native binaries, not the Python ML stack

An earlier version used a Python machine-learning stack (torch + onnxruntime +
`audio-separator`/UVR). It repeatedly failed **at runtime** with an opaque
`ImportError: import numpy failed` from onnxruntime — a failure that only
reproduced inside the deployment environment, never in local or build-time
runs, which made it effectively undebuggable.

This version keeps the AI quality but drops that fragility: the two models —
**DeepFilterNet3** (enhancement) and **Whisper** (transcription) — run as
self-contained native binaries with **no python, torch, onnxruntime, or numpy**.
There is nothing in the image that can throw `import numpy failed`. ffmpeg
handles extraction and amplification; DeepFilterNet does the heavy noise
removal; whisper.cpp does the captions.

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

## Limitations

- **CPU-only processing.** ffmpeg is fast, but very long clips still take time;
  there's a duration cap (`UVR_MAX_DURATION_SECONDS`) to keep jobs bounded.
- **Single instance, in-memory job state.** Don't scale horizontally without
  moving job state out of process.
- **Noise reduction, not source separation.** This lifts speech out of
  ambient/broadband noise; it does not separate overlapping speakers or strip
  musical accompaniment the way a dedicated separation model would.
- **Can't recover speech below the noise floor.** If a voice is quieter than
  the surrounding noise in the original recording, no amount of processing
  (this tool or otherwise) can reliably reconstruct the words — there's no
  signal left to lift. Pushing sliders to extremes on such audio tends to
  produce artifacts that Whisper can confidently mis-transcribe into
  plausible-sounding but entirely fabricated text; always sanity-check a
  transcript against the audio, especially at low confidence.
