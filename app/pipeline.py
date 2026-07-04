"""Audio processing pipeline for pulling faint speech out of loud noise.

Stages (each shells out to a self-contained CLI -- NO python ML stack, so
there is no torch/onnxruntime/numpy to fail at runtime):

1. extract    -- ffmpeg pulls the audio out of the source at 48kHz mono.
2. enhance    -- DeepFilterNet3 (deep-filter Rust binary) removes the
                 background noise while preserving speech, then an ffmpeg
                 denoise pass (strength set by the caller's TweakParams) tops
                 it up. Removing noise FIRST is what lets the next step boost
                 the voice without boosting the noise with it. Falls back to
                 ffmpeg-only denoise if the binary is absent.
3. amplify    -- ffmpeg lifts the now-clean quiet speech to a loud, clear,
                 consistent level (gate -> speechnorm -> compressor ->
                 loudnorm -> limiter), all tunable via TweakParams.
4. transcribe -- whisper.cpp reads the enhanced speech into text + SRT
                 captions. Best-effort: skipped if whisper isn't installed or
                 the caller turns it off.

TweakParams holds every user-adjustable knob (the "sliders"). All ffmpeg
filter values are clamped to ranges verified against this ffmpeg build so a
slider can never produce an invalid filter argument.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, fields
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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class TweakParams:
    """User-adjustable processing knobs, 0-100 "slider" scale unless noted.

    Defaults reproduce the original fixed pipeline's behavior.
    """

    noise_reduction: float = 55.0   # ffmpeg denoise strength (on top of AI)
    low_cut_hz: float = 90.0        # highpass cutoff, Hz (20-300)
    high_cut_hz: float = 7500.0     # lowpass cutoff, Hz (2000-8000)
    eq_freq: float = 2500.0         # voice-band EQ center, Hz (300-6000)
    eq_gain: float = 0.0            # voice-band EQ boost/cut, dB (-15..15); 0 = off
    notch_freq: float = 0.0         # tonal-noise notch center, Hz (0 = off, else 50-3000)
    notch_db: float = -18.0         # notch depth, dB (-40..0)
    vocal_boost: float = 100.0      # speechnorm/compressor makeup intensity
    compression: float = 42.0       # acompressor ratio intensity
    gain_db: float = 0.0            # extra loudness target shift, dB (-12..12)
    gate_threshold: float = -60.0   # noise gate threshold, dB (-70..-20); -70 ~= off
    use_ai_denoise: bool = True     # run DeepFilterNet3 if available
    df_postfilter: bool = False     # DeepFilterNet post-filter (--pf), stronger suppression
    df_strength: float = 100.0      # DeepFilterNet max attenuation, dB (0-100); 100 = default/full
    use_transcription: bool = True  # run whisper.cpp if available

    @classmethod
    def from_dict(cls, data: dict) -> "TweakParams":
        known = {f.name for f in fields(cls)}
        clean = {}
        for k, v in data.items():
            if k not in known or v is None:
                continue
            f = next(f for f in fields(cls) if f.name == k)
            clean[k] = bool(v) if f.type == "bool" else float(v)
        return cls(**clean)

    def clamped(self) -> "TweakParams":
        # notch_freq of 0 means "off"; otherwise keep it in a sane range.
        notch_freq = self.notch_freq if self.notch_freq <= 0 else _clamp(self.notch_freq, 50, 3000)
        return TweakParams(
            noise_reduction=_clamp(self.noise_reduction, 0, 100),
            low_cut_hz=_clamp(self.low_cut_hz, 20, 300),
            high_cut_hz=_clamp(self.high_cut_hz, 2000, 8000),
            eq_freq=_clamp(self.eq_freq, 300, 6000),
            eq_gain=_clamp(self.eq_gain, -15, 15),
            notch_freq=notch_freq,
            notch_db=_clamp(self.notch_db, -40, 0),
            vocal_boost=_clamp(self.vocal_boost, 0, 100),
            compression=_clamp(self.compression, 0, 100),
            gain_db=_clamp(self.gain_db, -12, 12),
            gate_threshold=_clamp(self.gate_threshold, -70, -20),
            use_ai_denoise=self.use_ai_denoise,
            df_postfilter=self.df_postfilter,
            df_strength=_clamp(self.df_strength, 0, 100),
            use_transcription=self.use_transcription,
        )


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


def extract_audio(src_path: Path, out_wav: Path, max_seconds: Optional[float] = None) -> Path:
    """Extract audio at 48kHz / 24-bit / mono. If max_seconds is set, only
    that much of the source is extracted (used for fast preview renders)."""
    cmd = ["ffmpeg", "-y"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += [
        "-i", str(src_path),
        "-vn", "-ac", "1", "-ar", str(WORK_SR), "-c:a", "pcm_s24le",
        str(out_wav),
    ]
    _run(cmd)
    return out_wav


def _ffmpeg_af(in_wav: Path, out_wav: Path, chain: str, sample_fmt: str = "pcm_s24le") -> Path:
    _run([
        "ffmpeg", "-y", "-i", str(in_wav),
        "-af", chain, "-ar", str(WORK_SR), "-c:a", sample_fmt, str(out_wav),
    ])
    return out_wav


def _denoise_chain(params: TweakParams) -> str:
    """Build the ffmpeg denoise/shaping segment.

    - highpass/lowpass: band-limit to the voice range.
    - notch (optional): a narrow deep cut at notch_freq to kill a steady tonal
      noise (engine whine, hum) -- only added when notch_freq > 0.
    - afftdn/anlmdn: broadband denoise; strength from noise_reduction.
    - voice EQ (optional): a peaking boost/cut at eq_freq to lift the band the
      voices sit in -- only added when eq_gain != 0.
    """
    nr = params.noise_reduction / 100.0
    nf = -20 - nr * 45          # 0 -> -20dB, 100 -> -65dB
    anlmdn_s = 0.0001 + nr * 0.0019   # 0 -> 0.0001, 100 -> 0.002

    parts = [
        f"highpass=f={params.low_cut_hz:.0f}",
        f"lowpass=f={params.high_cut_hz:.0f}",
    ]
    if params.notch_freq and params.notch_freq > 0 and params.notch_db < 0:
        # narrow peaking cut = surgical notch with tunable depth
        parts.append(f"equalizer=f={params.notch_freq:.0f}:t=q:w=8:g={params.notch_db:.1f}")
    parts.append(f"afftdn=nf={nf:.1f}:nt=w:tn=1")
    parts.append(f"anlmdn=s={anlmdn_s:.5f}:p=0.002:m=15")
    if abs(params.eq_gain) > 0.01:
        parts.append(f"equalizer=f={params.eq_freq:.0f}:t=q:w=1.4:g={params.eq_gain:.1f}")
    return ",".join(parts)


def enhance_speech(in_wav: Path, work_dir: Path, params: TweakParams, progress_cb: ProgressCB = None) -> Path:
    """Remove background noise: DeepFilterNet3 (if enabled + available) then
    an ffmpeg denoise pass whose strength is set by params.noise_reduction."""
    if progress_cb:
        progress_cb("removing background noise", 0.1)

    source = in_wav
    if params.use_ai_denoise and config.deepfilter_available():
        out_dir = work_dir / "dfn"
        out_dir.mkdir(parents=True, exist_ok=True)
        # deep-filter reads/writes 48kHz and writes <out_dir>/<input name>.
        base_cmd = [config.DEEPFILTER_BIN, "-o", str(out_dir)]
        extra = []
        if params.df_postfilter:
            extra.append("--pf")
        if params.df_strength < 100:
            # cap how much attenuation the model may apply (gentler at lower values)
            extra += ["--atten-lim", f"{params.df_strength:.0f}"]
        try:
            _run(base_cmd + extra + [str(in_wav)], timeout=1800)
        except PipelineError:
            # A bad/unsupported flag must never break the working AI denoise:
            # fall back to the plain, known-good invocation.
            if extra:
                logger.warning("deep-filter extra flags %s failed; retrying without them", extra)
                _run(base_cmd + [str(in_wav)], timeout=1800)
            else:
                raise
        enhanced = out_dir / in_wav.name
        if not enhanced.exists():
            cands = list(out_dir.glob("*.wav"))
            if cands:
                enhanced = cands[0]
        if enhanced.exists():
            source = enhanced
        else:
            logger.warning("DeepFilterNet produced no output; continuing with ffmpeg-only denoise")
    elif params.use_ai_denoise:
        logger.warning("AI denoise requested but DeepFilterNet binary not found at %s", config.DEEPFILTER_BIN)

    if progress_cb:
        progress_cb("removing background noise", 0.5)

    out = work_dir / "denoised.wav"
    _ffmpeg_af(source, out, _denoise_chain(params))

    if progress_cb:
        progress_cb("background noise removed", 1.0)
    return out


def _amplify_chain(params: TweakParams) -> str:
    """Build the ffmpeg amplification segment.

    - agate: noise gate, cuts residual hiss below the threshold between words.
    - speechnorm: raises quiet syllables (e valid range is 1-50 in this
      ffmpeg build; vocal_boost 0-100 maps onto it).
    - acompressor: compresses remaining dynamics; ratio (1-20 valid) and
      makeup gain scale with compression/vocal_boost.
    - loudnorm: EBU R128 normalize to a loud target, shiftable by gain_db.
    - alimiter: catches peaks so nothing clips.
    """
    e = _clamp(1 + (params.vocal_boost / 100.0) * 49, 1, 50)
    ratio = _clamp(1 + (params.compression / 100.0) * 14, 1, 20)
    makeup = _clamp(1 + (params.vocal_boost / 100.0) * 7, 1, 16)
    target_i = _clamp(-13 + params.gain_db, -40, -5)

    return (
        f"agate=threshold={params.gate_threshold:.0f}dB:ratio=6:attack=5:release=100,"
        f"speechnorm=e={e:.1f}:r=0.0003:p=0.55,"
        f"acompressor=threshold=-24dB:ratio={ratio:.1f}:attack=8:release=180:makeup={makeup:.1f},"
        f"dynaudnorm=f=150:g=15:p=0.9:m=40,"
        f"loudnorm=I={target_i:.1f}:TP=-1.0:LRA=11,"
        f"alimiter=limit=0.97"
    )


def amplify_voice(in_wav: Path, out_wav: Path, out_mp3: Optional[Path], params: TweakParams,
                   progress_cb: ProgressCB = None) -> None:
    """Lift the now-clean quiet speech to a loud, clear, consistent level."""
    if progress_cb:
        progress_cb("amplifying quiet speech", 0.1)

    _ffmpeg_af(in_wav, out_wav, _amplify_chain(params))

    if out_mp3 is not None:
        if progress_cb:
            progress_cb("encoding downloadable copy", 0.8)
        _run([
            "ffmpeg", "-y", "-i", str(out_wav),
            "-c:a", "libmp3lame", "-q:a", "1", str(out_mp3),
        ])
    if progress_cb:
        progress_cb("done", 1.0)


def render_preview(src_path: Path, work_dir: Path, params: TweakParams, seconds: float = 20.0) -> Path:
    """Render only the first `seconds` of the source through the real
    enhance+amplify chain (optionally with AI denoise), for fast auditioning
    of slider settings without processing the whole file. Returns an MP3."""
    work_dir.mkdir(parents=True, exist_ok=True)
    raw = work_dir / "preview_src.wav"
    extract_audio(src_path, raw, max_seconds=seconds)

    enhanced = enhance_speech(raw, work_dir, params)
    final_wav = work_dir / "preview_out.wav"
    final_mp3 = work_dir / "preview_out.mp3"
    amplify_voice(enhanced, final_wav, final_mp3, params)
    return final_mp3


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
