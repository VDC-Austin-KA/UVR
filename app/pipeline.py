"""Audio processing pipeline.

Video in -> lossless audio extraction -> UVR-based vocal isolation
(separates speech from music/ambience/traffic) -> DeepFilterNet speech
denoising (kills residual broadband/traffic noise) -> ffmpeg mastering
chain that boosts quiet passages so faint speech becomes intelligible
without clipping the loud parts.

Every stage is a plain function that shells out to ffmpeg or calls a
local model; nothing here talks to the job manager directly so it can
be unit-tested or reused from a CLI.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app import config

logger = logging.getLogger("uvr.pipeline")

ProgressCB = Optional[Callable[[str, float], None]]

# Sample rate DeepFilterNet's bundled checkpoints expect.
DENOISE_SR = 48000


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
    """Pull the audio out of the source video at the highest practical
    fidelity: 48kHz / 24-bit PCM, stereo preserved."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(DENOISE_SR),
        "-c:a",
        "pcm_s24le",
        str(out_wav),
    ]
    _run(cmd)
    return out_wav


def separate_vocals(wav_path: Path, out_dir: Path, progress_cb: ProgressCB = None) -> Path:
    """Run a UVR (Ultimate Vocal Remover) MDX-Net model to split the
    speech/vocal content away from music, ambience, and other
    background sound (e.g. traffic)."""
    # Imported lazily: this pulls in torch/onnxruntime, which are heavy
    # and unnecessary for code paths that never separate audio (tests, etc).
    #
    # Import numpy directly first. onnxruntime (pulled in transitively) reports
    # *any* numpy load failure as an opaque "ImportError: import numpy failed",
    # which hides the real cause. Importing numpy here surfaces the true error
    # (missing shared lib, unsupported CPU, thread/allocation failure, ABI
    # mismatch) both in the logs and in the message shown to the user.
    try:
        import numpy  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        raise PipelineError(
            "NumPy failed to load in the processing environment "
            f"({type(exc).__name__}: {exc}). This is an environment/runtime "
            "issue, not a problem with your file."
        ) from exc
    from audio_separator.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("loading separation model", 0.0)

    separator = Separator(
        output_dir=str(out_dir),
        output_format="WAV",
        model_file_dir=str(config.MODEL_DIR),
        output_single_stem="Vocals",
        normalization_threshold=0.9,
        log_level=logging.WARNING,
    )
    separator.load_model(model_filename=config.SEPARATION_MODEL)

    if progress_cb:
        progress_cb("isolating voices", 0.3)

    output_files = separator.separate(str(wav_path))
    if not output_files:
        raise PipelineError("Vocal isolation produced no output.")

    vocals_path = Path(output_files[0])
    if not vocals_path.is_absolute():
        vocals_path = out_dir / vocals_path.name
    if not vocals_path.exists():
        raise PipelineError("Vocal isolation output file went missing.")

    if progress_cb:
        progress_cb("voices isolated", 1.0)

    return vocals_path


def _resample_mono(in_wav: Path, out_wav: Path, sr: int) -> Path:
    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ])
    return out_wav


def denoise_speech(vocals_wav: Path, work_dir: Path, progress_cb: ProgressCB = None) -> Path:
    """Run DeepFilterNet (standalone Rust CLI, model weights embedded in
    the binary) over the isolated vocal stem to strip residual
    broadband/environmental noise (traffic, wind, hiss, room rumble)
    while preserving speech."""
    if progress_cb:
        progress_cb("preparing audio for denoising", 0.0)

    mono_input = _resample_mono(vocals_wav, work_dir / "vocals_48k_mono.wav", DENOISE_SR)

    if progress_cb:
        progress_cb("removing background noise", 0.3)

    out_dir = work_dir / "denoised"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([config.DEEPFILTER_BIN, "--pf", "-o", str(out_dir), str(mono_input)])

    out_path = out_dir / mono_input.name
    if not out_path.exists():
        candidates = list(out_dir.glob("*.wav"))
        if not candidates:
            raise PipelineError("Denoising produced no output.")
        out_path = candidates[0]

    if progress_cb:
        progress_cb("noise removed", 1.0)

    return out_path


def master_speech(in_wav: Path, out_wav: Path, out_mp3: Path, progress_cb: ProgressCB = None) -> None:
    """Bring quiet, barely-audible speech up to an intelligible,
    consistent level without blowing out louder passages, then emit a
    lossless master plus a compressed copy for quick mobile playback.

    - highpass: shaves off sub-100Hz rumble (engine/traffic noise lives here)
    - afftdn: a second, gentler FFT denoise pass on top of DeepFilterNet
    - speechnorm: boosts quiet syllables toward audibility (this is the
      "amplify the barely-hearable voice" step)
    - loudnorm: final EBU R128 loudness normalization for a consistent,
      comfortable listening level
    """
    if progress_cb:
        progress_cb("boosting quiet speech", 0.0)

    filter_chain = (
        "highpass=f=90,"
        "afftdn=nf=-25,"
        "speechnorm=e=12.5:r=0.0001:p=0.95,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )

    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-af",
        filter_chain,
        "-ar",
        str(DENOISE_SR),
        "-c:a",
        "pcm_s24le",
        str(out_wav),
    ])

    if progress_cb:
        progress_cb("encoding downloadable copy", 0.7)

    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(out_wav),
        "-c:a",
        "libmp3lame",
        "-q:a",
        "0",
        str(out_mp3),
    ])

    if progress_cb:
        progress_cb("done", 1.0)
