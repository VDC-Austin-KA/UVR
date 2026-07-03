"""Audio processing pipeline for pulling faint speech out of loud noise.

Stages (each shells out to a self-contained CLI -- NO python ML stack, so
there is no torch/onnxruntime/numpy to fail at runtime):

1. extract    -- ffmpeg pulls the audio out of the source at 48kHz mono.
2. enhance    -- DeepFilterNet3 (deep-filter Rust binary) removes the
                 background noise while preserving speech. This is the heavy
                 lifting; ffmpeg denoisers are a fallback if the binary is
                 absent. Removing noise FIRST is what lets the next step boost
                 the voice without boosting the noise with it.
3. amplify    -- ffmpeg lifts the now-clean quiet speech to a loud, clear,
                 consistent level (speechnorm -> compressor -> loudnorm ->
                 limiter).
4. transcribe -- whisper.cpp reads the enhanced speech into text + SRT
                 captions. Best-effort: skipped if whisper isn't installed.
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

WORK_SR = 48000        # main working rate
WHISPER_SR = 16000     # whisper.cpp expects 16kHz mono


class PipelineError(RuntimeError):
    """Raised for expected, user-facing failures (bad file, too long, etc.)."""


@dataclass
class SourceInfo:
    duration_seconds: float
    has_audio: bool


def _run(cmd: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    logger.info("running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error("command failed (%s): %s", cmd[0], result.stderr[-4000:])
        raise PipelineError(f"{cmd[0]} failed: {result.stderr.strip()[-800:]}")
    return result


def probe_source(path: Path) -> SourceInfo:
    """Read duration + whether an audio stream exists, via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
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
    """Extract audio at 48kHz / 24-bit / mono."""
    _run([
        "ffmpeg", "-y", "-i", str(src_path),
        "-vn", "-ac", "1", "-ar", str(WORK_SR), "-c:a", "pcm_s24le",
        str(out_wav),
    ])
    return out_wav


def _ffmpeg_af(in_wav: Path, out_wav: Path, chain: str, sample_fmt: str = "pcm_s24le") -> Path:
    _run([
        "ffmpeg", "-y", "-i", str(in_wav),
        "-af", chain, "-ar", str(WORK_SR), "-c:a", sample_fmt, str(out_wav),
    ])
    return out_wav


def enhance_speech(in_wav: Path, work_dir: Path, progress_cb: ProgressCB = None) -> Path:
    """Remove background noise with DeepFilterNet3 (falls back to ffmpeg).

    DeepFilterNet is a deep-learning speech enhancer that predicts a
    suppression filter per frequency bin -- far better than ffmpeg's
    generic denoisers at non-stationary noise like traffic/crowd. We run it
    twice for stronger suppression on very noisy sources.
    """
    if progress_cb:
        progress_cb("removing background noise", 0.1)

    if config.deepfilter_available():
        out_dir = work_dir / "dfn"
        out_dir.mkdir(parents=True, exist_ok=True)
        # deep-filter reads/writes 48kHz and writes <out_dir>/<input name>.
        _run([config.DEEPFILTER_BIN, "-o", str(out_dir), str(in_wav)], timeout=1800)
        enhanced = out_dir / in_wav.name
        if not enhanced.exists():
            cands = list(out_dir.glob("*.wav"))
            if not cands:
                raise PipelineError("Speech enhancement produced no output.")
            enhanced = cands[0]
        if progress_cb:
            progress_cb("background noise removed", 1.0)
        return enhanced

    # Fallback: ffmpeg-only denoise (DeepFilterNet binary unavailable).
    logger.warning("DeepFilterNet binary not found at %s; using ffmpeg denoise", config.DEEPFILTER_BIN)
    out = work_dir / "denoised_ff.wav"
    _ffmpeg_af(
        in_wav, out,
        "highpass=f=80,lowpass=f=7500,afftdn=nf=-28:nt=w:tn=1,afftdn=nf=-24:nt=w:tn=1,anlmdn=s=0.0006:p=0.002:m=15",
    )
    if progress_cb:
        progress_cb("background noise removed", 1.0)
    return out


def amplify_voice(in_wav: Path, out_wav: Path, out_mp3: Path, progress_cb: ProgressCB = None) -> None:
    """Lift the now-clean quiet speech to a loud, clear, consistent level.

    - speechnorm: raise quiet syllables toward the target (main boost).
    - acompressor: compress remaining dynamics so faint words come up to the
      level of louder ones.
    - loudnorm: EBU R128 normalize to a loud target (-13 LUFS).
    - alimiter: catch peaks so nothing clips.
    """
    if progress_cb:
        progress_cb("amplifying quiet speech", 0.1)

    chain = (
        "highpass=f=70,"
        "speechnorm=e=50:r=0.0003:p=0.55,"
        "acompressor=threshold=-24dB:ratio=6:attack=8:release=180:makeup=8,"
        "dynaudnorm=f=150:g=15:p=0.9:m=40,"
        "loudnorm=I=-13:TP=-1.0:LRA=11,"
        "alimiter=limit=0.97"
    )
    _ffmpeg_af(in_wav, out_wav, chain)

    if progress_cb:
        progress_cb("encoding downloadable copy", 0.8)

    _run([
        "ffmpeg", "-y", "-i", str(out_wav),
        "-c:a", "libmp3lame", "-q:a", "1", str(out_mp3),
    ])
    if progress_cb:
        progress_cb("done", 1.0)


def transcribe(in_wav: Path, work_dir: Path, progress_cb: ProgressCB = None) -> Optional[dict]:
    """Transcribe the enhanced speech with whisper.cpp -> text + SRT.

    Returns {"txt": Path, "srt": Path, "text": str} or None if whisper is
    unavailable or produced nothing. Never raises: transcription is a bonus
    on top of the (already delivered) cleaned audio.
    """
    if not config.whisper_available():
        logger.warning("whisper.cpp unavailable (bin=%s model=%s); skipping transcription",
                       config.WHISPER_BIN, config.WHISPER_MODEL)
        return None

    if progress_cb:
        progress_cb("transcribing speech", 0.1)

    try:
        wav16 = work_dir / "for_whisper_16k.wav"
        _run([
            "ffmpeg", "-y", "-i", str(in_wav),
            "-ac", "1", "-ar", str(WHISPER_SR), "-c:a", "pcm_s16le", str(wav16),
        ])

        out_prefix = work_dir / "transcript"
        _run([
            config.WHISPER_BIN,
            "-m", config.WHISPER_MODEL,
            "-f", str(wav16),
            "-otxt", "-osrt",
            "-of", str(out_prefix),
            "-nt",
        ], timeout=3600)

        txt = out_prefix.with_suffix(".txt")
        srt = out_prefix.with_suffix(".srt")
        text = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
        if progress_cb:
            progress_cb("transcription done", 1.0)
        if not text:
            return None
        return {"txt": txt, "srt": srt, "text": text}
    except Exception:
        logger.exception("transcription failed (non-fatal)")
        return None
