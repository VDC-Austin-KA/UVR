"""Audio processing pipeline.

Video/audio in -> lossless audio extraction -> a speech-focused ffmpeg
cleanup chain that (1) knocks down steady background noise (traffic,
wind, hiss, hum), (2) restricts the signal to the voice band, and
(3) heavily amplifies quiet speech so barely-audible voices become
clearly hearable, then normalizes to a consistent, comfortable level.

Everything here shells out to ffmpeg -- there is deliberately NO Python
machine-learning stack (no torch / onnxruntime / numpy). That stack was
the source of persistent "import numpy failed" runtime crashes on the
host, and ffmpeg's built-in denoisers + dynamics do the job with zero
heavyweight dependencies, so the service is small, fast, and reliable.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("uvr.pipeline")

ProgressCB = Optional[Callable[[str, float], None]]

# Working sample rate. 48kHz keeps full speech bandwidth with headroom.
WORK_SR = 48000


class PipelineError(RuntimeError):
    """Raised for expected, user-facing failures (bad file, too long, etc.)."""


@dataclass
class SourceInfo:
    duration_seconds: float
    has_audio: bool


def _run(cmd: list[str]) -> None:
    logger.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("command failed (%s): %s", cmd[0], result.stderr[-4000:])
        raise PipelineError(f"{cmd[0]} failed: {result.stderr.strip()[-800:]}")


def probe_source(path: Path) -> SourceInfo:
    """Read duration + whether an audio stream exists, via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError("Could not read that file — is it a valid video/audio file?")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_audio:
        raise PipelineError("No audio track found in the uploaded file.")

    duration = 0.0
    fmt_duration = data.get("format", {}).get("duration")
    if fmt_duration is not None:
        duration = float(fmt_duration)
    else:
        for s in streams:
            if s.get("duration"):
                duration = max(duration, float(s["duration"]))

    return SourceInfo(duration_seconds=duration, has_audio=has_audio)


def extract_audio(src_path: Path, out_wav: Path) -> Path:
    """Pull the audio out of the source at high fidelity: 48kHz / 24-bit
    PCM, downmixed to mono (voice work is mono; it also gives the noise
    filters a single coherent channel to operate on)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(WORK_SR),
        "-c:a",
        "pcm_s24le",
        str(out_wav),
    ]
    _run(cmd)
    return out_wav


def _filtered(in_wav: Path, out_wav: Path, filter_chain: str, sample_fmt: str = "pcm_s24le") -> Path:
    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-af",
        filter_chain,
        "-ar",
        str(WORK_SR),
        "-c:a",
        sample_fmt,
        str(out_wav),
    ])
    return out_wav


def reduce_noise(in_wav: Path, out_wav: Path, progress_cb: ProgressCB = None) -> Path:
    """Suppress steady background noise and confine the signal to the
    voice band.

    - highpass f=80 / lowpass f=7500: drops sub-bass rumble (engines,
      traffic, wind, HVAC hum) and high hiss that carry no speech.
    - afftdn (two passes, noise-tracking): adaptive FFT denoiser that
      learns the background noise profile and subtracts it -- the main
      workhorse for "reduce the sound around the voice".
    - anlmdn: a gentler non-local-means pass to clean up what's left
      without chewing into the speech.
    """
    if progress_cb:
        progress_cb("removing background noise", 0.1)

    chain = (
        "highpass=f=80,"
        "lowpass=f=7500,"
        "afftdn=nf=-28:nt=w:tn=1,"
        "afftdn=nf=-24:nt=w:tn=1,"
        "anlmdn=s=0.0006:p=0.002:m=15"
    )
    _filtered(in_wav, out_wav, chain)

    if progress_cb:
        progress_cb("background noise reduced", 1.0)
    return out_wav


def amplify_voice(in_wav: Path, out_wav: Path, out_mp3: Path, progress_cb: ProgressCB = None) -> None:
    """Make faint speech clearly audible without blowing out the loud
    parts, then emit a lossless master + a compressed copy.

    - speechnorm: raises quiet syllables toward a target level -- this is
      the "amplify the barely-hearable voice" step (aggressive expansion).
    - dynaudnorm: smooths overall loudness over time so nothing stays too
      quiet or spikes too loud across the clip.
    - loudnorm: final EBU R128 pass for a consistent, comfortable level.
    - alimiter: catches any peaks so the boosted output never clips.
    """
    if progress_cb:
        progress_cb("amplifying quiet speech", 0.1)

    chain = (
        "speechnorm=e=50:r=0.0004:p=0.6:c=15,"
        "dynaudnorm=f=150:g=15:p=0.9:m=30,"
        "loudnorm=I=-14:TP=-1.5:LRA=11,"
        "alimiter=limit=0.95"
    )
    _filtered(in_wav, out_wav, chain)

    if progress_cb:
        progress_cb("encoding downloadable copy", 0.8)

    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(out_wav),
        "-c:a",
        "libmp3lame",
        "-q:a",
        "1",
        str(out_mp3),
    ])

    if progress_cb:
        progress_cb("done", 1.0)
