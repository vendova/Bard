#!/usr/bin/env python3
"""Post-processing operations for BeatSync-Engine.

Five areas, each with a fully FFmpeg-native / numpy path and, where it makes
sense, an optional heavier backend:

1. Smoothing   - `stabilize_video` (vid.stab), `interpolate_fps` (minterpolate),
                 plus an optional neural RIFE adapter (`rife_available`).
2. Clipping    - `detect_silences` (silencedetect) and `active_segments`, used
                 to skip dead space in source clips.
3. Syncing     - `estimate_offset` / `align_offset` cross-correlate two audio
                 tracks (multicam sync).
4. Interchange - `write_edl`, `write_cutlist_json`, `write_otio` export the
                 recovered/planned timeline.
5. Glue        - a single `postprocess` entry point the render path can call.

Everything degrades gracefully: ffmpeg filters are probed before use and the
OpenTimelineIO export is optional (a clear error is raised if it is missing).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:  # package-relative when imported as ``forensics.post_ops`` style
    from .media_paths import resolve_ffmpeg
except ImportError:  # ``src`` on sys.path (the app's flat import style)
    from media_paths import resolve_ffmpeg

FFMPEG, FFPROBE = resolve_ffmpeg()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _run(cmd: Sequence[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True,
                          timeout=timeout, check=False)


def _run_checked(cmd: Sequence[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    res = _run(cmd, timeout=timeout)
    if res.returncode != 0:
        tail = "\n".join((res.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(
            f"{os.path.basename(cmd[0])} failed (code {res.returncode}):\n{tail}")
    return res


_FILTER_CACHE: Optional[set] = None


def available_filters() -> set:
    """Names of FFmpeg filters this build ships (cached)."""
    global _FILTER_CACHE
    if _FILTER_CACHE is None:
        res = _run([FFMPEG, "-hide_banner", "-filters"], timeout=30)
        names = set()
        for line in (res.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 3 and "->" in parts[2]:
                names.add(parts[1])
        _FILTER_CACHE = names
    return _FILTER_CACHE


def has_filter(name: str) -> bool:
    return name in available_filters()


def probe_fps(path: str, default: float = 30.0) -> float:
    res = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1", path], timeout=30)
    raw = (res.stdout or "").strip()
    try:
        if "/" in raw:
            num, den = raw.split("/")
            return float(num) / float(den)
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return default


def probe_duration(path: str, default: float = 0.0) -> float:
    res = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path], timeout=30)
    try:
        return float((res.stdout or "").strip())
    except ValueError:
        return default


def _enc_args(quality: str) -> List[str]:
    if quality == "lossless":
        return ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast"]
    if quality == "draft":
        return ["-c:v", "libx264", "-crf", "30", "-preset", "ultrafast"]
    return ["-c:v", "libx264", "-crf", os.environ.get("BEATSYNC_CRF", "18"),
            "-preset", os.environ.get("BEATSYNC_PRESET", "medium")]


# ---------------------------------------------------------------------------
# 1. Smoothing
# ---------------------------------------------------------------------------

def stabilize_video(src: str, dst: str, *, smoothing: int = 10, zoom: int = 0,
                    optzoom: int = 1, tripod: bool = False, shake_detect: int = 5,
                    quality: str = "high", timeout: int = 3600) -> bool:
    """Two-pass vid.stab stabilization. Returns True on success."""
    if not (has_filter("vidstabdetect") and has_filter("vidstabtransform")):
        raise RuntimeError("ffmpeg build lacks vid.stab (vidstabdetect/vidstabtransform)")

    transforms = tempfile.NamedTemporaryFile(suffix=".trf", delete=False).name
    try:
        _run_checked([
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
            "-vf", f"vidstabdetect=shakiness={shake_detect}:result={transforms}",
            "-f", "null", "-",
        ], timeout=timeout)
        tripod_arg = "tripod=1" if tripod else ""
        vf = (f"vidstabtransform=input={transforms}:smoothing={smoothing}"
              f":zoom={zoom}:optzoom={optzoom}:interpol=bicubic"
              + (f":{tripod_arg}" if tripod_arg else ""))
        _run_checked([
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
            "-vf", vf, *_enc_args(quality), "-c:a", "copy", dst,
        ], timeout=timeout)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    finally:
        if os.path.exists(transforms):
            try:
                os.remove(transforms)
            except OSError:
                pass


def interpolate_fps(src: str, dst: str, fps: float, *, mi_mode: str = "mci",
                    me_mode: str = "bidir", me_algorithm: str = "epzs",
                    vsbmc: int = 0, quality: str = "high",
                    timeout: int = 7200) -> bool:
    """Frame-rate conversion via FFmpeg's motion-compensated ``minterpolate``."""
    if not has_filter("minterpolate"):
        raise RuntimeError("ffmpeg build lacks the minterpolate filter")
    vf = (f"minterpolate=fps={fps}:mi_mode={mi_mode}:me_mode={me_mode}"
          f":me={me_algorithm}:vsbmc={vsbmc}")
    _run_checked([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-vf", vf, *_enc_args(quality), "-c:a", "copy", dst,
    ], timeout=timeout)
    return os.path.exists(dst) and os.path.getsize(dst) > 0


def rife_available() -> bool:
    """Whether the optional neural RIFE backend can run here.

    RIFE needs PyTorch/ONNX plus downloaded weights, which the portable CPU
    build deliberately omits. This probes without importing torch at module
    load so the app stays import-safe everywhere.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return bool(os.environ.get("BEATSYNC_RIFE_PATH"))


def interpolate_rife(src: str, dst: str, multiplier: int = 2,
                     rife_path: Optional[str] = None) -> bool:
    """Neural interpolation via an external Practical-RIFE checkout.

    Kept deliberately isolated: it shells out to the RIFE repo's
    ``inference_video.py`` if (and only if) one is configured. Without torch +
    a checkout this raises, so callers should guard with :func:`rife_available`.
    """
    repo = rife_path or os.environ.get("BEATSYNC_RIFE_PATH", "")
    if not repo or not os.path.exists(os.path.join(repo, "inference_video.py")):
        raise RuntimeError("RIFE checkout not found (set BEATSYNC_RIFE_PATH)")
    _run_checked(["python", os.path.join(repo, "inference_video.py"),
                  f"--video={src}", f"--multi={multiplier}",
                  f"--output={dst}"], timeout=7200)
    return os.path.exists(dst) and os.path.getsize(dst) > 0


# ---------------------------------------------------------------------------
# 2. Clipping / dead-space
# ---------------------------------------------------------------------------

@dataclass
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_silences(src: str, *, noise_db: float = -30.0,
                    min_duration: float = 0.5, timeout: int = 600) -> List[Silence]:
    """Silence intervals via FFmpeg's ``silencedetect`` audio filter."""
    res = _run([FFMPEG, "-hide_banner", "-nostats", "-i", src,
                "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
                "-f", "null", "-"], timeout=timeout)
    silences: List[Silence] = []
    pending_start: Optional[float] = None
    for line in (res.stderr or "").splitlines():
        if "silence_start:" in line:
            try:
                pending_start = float(line.split("silence_start:")[1].split()[0])
            except (IndexError, ValueError):
                pending_start = None
        elif "silence_end:" in line and pending_start is not None:
            try:
                end = float(line.split("silence_end:")[1].split()[0])
            except (IndexError, ValueError):
                continue
            silences.append(Silence(pending_start, end))
            pending_start = None
    return silences


def active_segments(duration: float, silences: Sequence[Silence],
                    min_active: float = 0.3) -> List[Tuple[float, float]]:
    """Invert silences into the loud/active intervals worth sampling."""
    segments: List[Tuple[float, float]] = []
    cursor = 0.0
    for s in sorted(silences, key=lambda x: x.start):
        if s.start - cursor >= min_active:
            segments.append((cursor, s.start))
        cursor = max(cursor, s.end)
    if duration - cursor >= min_active:
        segments.append((cursor, duration))
    return segments


def has_audio_stream(path: str) -> bool:
    res = _run([FFPROBE, "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", path], timeout=30)
    return bool((res.stdout or "").strip())


def trim_to_active_segments(src: str, dst: str,
                            segments: Sequence[Tuple[float, float]],
                            *, quality: str = "draft", timeout: int = 7200) -> bool:
    """Write a copy of *src* containing only the given active intervals.

    Keeps video and audio in sync by trimming both streams per segment and
    concatenating them. Returns False (without raising) if the source has no
    usable audio or the graph fails, so callers can fall back to the original.
    """
    if not segments:
        return False
    if not has_audio_stream(src):
        return False

    parts = []
    concat_inputs = []
    for i, (a, b) in enumerate(segments):
        dur = max(0.01, b - a)
        parts.append(f"[0:v]trim=start={a:.4f}:duration={dur:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.4f}:duration={dur:.4f},asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")
    graph = (";".join(parts) + ";" + "".join(concat_inputs)
             + f"concat=n={len(segments)}:v=1:a=1[outv][outa]")
    res = _run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                "-filter_complex", graph, "-map", "[outv]", "-map", "[outa]",
                *_enc_args(quality), "-c:a", "aac", dst], timeout=timeout)
    return res.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0


# ---------------------------------------------------------------------------
# 3. Syncing (audio cross-correlation)
# ---------------------------------------------------------------------------

def _extract_mono_wav(src: str, sr: int = 16000, max_seconds: float = 300.0) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
           "-vn"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-ac", "1", "-ar", str(sr), out]
    _run_checked(cmd, timeout=600)
    return out


def _envelope(path: str, sr: int = 16000):
    """Onset-strength envelope (cheap, gain-invariant) for correlation."""
    import numpy as np
    import soundfile as sf
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    # One-pole RMS envelope at ~50 Hz rather than full spectral onset to keep
    # this dependency-light and fast; it is gain-invariant after normalising.
    hop = max(1, rate // 50)
    n = len(data) // hop
    if n == 0:
        return np.zeros(1, dtype="float32")
    frames = data[:n * hop].reshape(n, hop)
    env = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    env = env - env.mean()
    std = env.std()
    return (env / std if std > 0 else env).astype("float32")


_ENV_RATE = 50.0  # envelope frames per second (see _envelope)


def _best_lag(ref, tgt, max_lag_samples: int):
    """Lag (in envelope frames) maximising the *normalised* overlap correlation.

    Positive lag means ``ref`` must shift right to line up with ``tgt``. Each
    candidate lag is normalised by the overlap length (and by both norms), so
    long-overlap lags do not win over the true peak the way a raw FFT
    cross-correlation does.
    """
    import numpy as np
    best_lag = 0
    best_score = -2.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            r = ref[lag:]
            t = tgt[:len(r)]
        else:
            t = tgt[-lag:]
            r = ref[:len(t)]
        n = min(len(r), len(t))
        if n < 8:
            continue
        r = r[:n] - r[:n].mean()
        t = t[:n] - t[:n].mean()
        denom = float(np.linalg.norm(r) * np.linalg.norm(t)) + 1e-9
        score = float(np.dot(r, t) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def estimate_offset(reference: str, target: str, *, max_lag: float = 5.0,
                    max_seconds: float = 300.0, sr: int = 16000) -> float:
    """Seconds by which *target*'s audio lags *reference*.

    Positive => target starts later; apply it as a leading delay to align.
    Cross-correlates the two normalised RMS envelopes.
    """
    ref_wav = _extract_mono_wav(reference, sr=sr, max_seconds=max_seconds)
    tgt_wav = _extract_mono_wav(target, sr=sr, max_seconds=max_seconds)
    try:
        ref = _envelope(ref_wav, sr=sr)
        tgt = _envelope(tgt_wav, sr=sr)
        if len(ref) < 8 or len(tgt) < 8:
            return 0.0
        max_lag_samples = int(round(max_lag * _ENV_RATE))
        lag = _best_lag(ref, tgt, max_lag_samples)
        # tgt[j] ~= ref[j - d] for a delay of d frames => aligning shifts ref
        # left by d, i.e. lag = -d, so the delay is -lag.
        offset = -lag / _ENV_RATE
        return round(float(offset), 4)
    finally:
        for p in (ref_wav, tgt_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def align_offset(dst: str, src: str, offset_seconds: float,
                 quality: str = "high", timeout: int = 3600) -> bool:
    """Write *src* to *dst* with its audio shifted by *offset_seconds*.

    A positive offset delays the audio (``adelay``); a negative one advances it
    by trimming the leading silence-equivalent with ``atrim``.
    """
    if offset_seconds >= 0:
        ms = int(round(offset_seconds * 1000))
        af = f"adelay={ms}|{ms}"
    else:
        af = f"atrim=start={abs(offset_seconds):.4f},asetpts=PTS-STARTPTS"
    _run_checked([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                  "-af", af, "-c:v", "copy", *_enc_args(quality), dst],
                 timeout=timeout)
    return os.path.exists(dst) and os.path.getsize(dst) > 0


# ---------------------------------------------------------------------------
# 4. Interchange (timeline export)
# ---------------------------------------------------------------------------

@dataclass
class Edit:
    """One placed clip: a source range at a record-time position."""
    reel: str
    start_src: float
    end_src: float
    start_rec: float = 0.0
    name: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_src - self.start_src)


def _tc(seconds: float, fps: float) -> str:
    """Non-drop-frame timecode HH:MM:SS:FF."""
    fps_i = max(1, int(round(fps)))
    total = max(0, int(round(seconds * fps_i)))
    ff = total % fps_i
    total //= fps_i
    ss = total % 60
    mm = (total // 60) % 60
    hh = (total // 3600) % 24
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def write_edl(edits: Sequence[Edit], path: str, *, fps: float = 30.0,
              title: str = "BEATSYNC") -> str:
    """Write a CMX3600 EDL that any NLE (Premiere/Resolve/FCP) can import."""
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for i, e in enumerate(edits, 1):
        reel = (e.reel or "AX")[:8].upper().ljust(8)
        lines.append(
            f"{i:03d}  {reel} V     C        "
            f"{_tc(e.start_src, fps)} {_tc(e.end_src, fps)} "
            f"{_tc(e.start_rec, fps)} {_tc(e.start_rec + e.duration, fps)}")
        if e.name:
            lines.append(f"* FROM CLIP NAME: {e.name}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_cutlist_json(edits: Sequence[Edit], path: str, *, fps: float = 30.0,
                       meta: Optional[Dict] = None) -> str:
    payload = {
        "fps": fps,
        "edit_count": len(edits),
        "edits": [
            {"index": i, "reel": e.reel, "name": e.name,
             "source_in": round(e.start_src, 4), "source_out": round(e.end_src, 4),
             "record_in": round(e.start_rec, 4), "record_out": round(e.start_rec + e.duration, 4),
             "duration": round(e.duration, 4)}
            for i, e in enumerate(edits)
        ],
    }
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def write_otio(edits: Sequence[Edit], path: str, *, fps: float = 30.0,
               name: str = "BeatSync timeline") -> str:
    """Export the timeline as OpenTimelineIO (.otio), an optional dependency."""
    try:
        import opentimelineio as otio
    except Exception as exc:  # pragma: no cover - exercised only when absent
        raise RuntimeError(
            "OpenTimelineIO is not installed (pip install opentimelineio)") from exc

    rate = max(1, int(round(fps)))
    timeline = otio.schema.Timeline(name=name)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)
    for e in edits:
        media = otio.schema.ExternalReference(
            target_url=e.name or e.reel or "unknown",
            available_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(e.start_src, rate),
                otio.opentime.RationalTime(e.duration, rate)))
        clip = otio.schema.Clip(
            name=e.name or e.reel,
            media_reference=media,
            source_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(e.start_src, rate),
                otio.opentime.RationalTime(e.duration, rate)))
        track.append(clip)
    otio.adapters.write_to_file(timeline, path)
    return path


def edits_from_plan(planned: Sequence[Dict], *, fps: float = 30.0,
                    record_start: float = 0.0) -> List[Edit]:
    """Convert the render planner's clip dicts into :class:`Edit` records.

    Accepts the flexible shapes the planner uses (``source_file``/``path``,
    ``start``/``start_time``, ``duration``, optional ``record_start``).
    """
    edits: List[Edit] = []
    rec = record_start
    for item in planned:
        src = item.get("source_file") or item.get("path") or item.get("video_file") or ""
        start = float(item.get("start_time", item.get("start", 0.0)) or 0.0)
        dur = item.get("final_duration", item.get("duration"))
        if dur is None:
            dur = item.get("source_duration", 0.0)
        dur = float(dur or 0.0)
        if dur <= 0 and "end_time" in item:
            dur = float(item["end_time"]) - start
        rec_in = float(item.get("record_start", rec) or rec)
        edits.append(Edit(reel=os.path.basename(src), name=src,
                          start_src=start, end_src=start + dur,
                          start_rec=rec_in))
        rec = rec_in + dur
    return edits


# ---------------------------------------------------------------------------
# 5. Glue
# ---------------------------------------------------------------------------

@dataclass
class PostOps:
    """Render-path post-processing switches (all opt-in)."""
    stabilize: bool = False
    stabilization_smoothing: int = 10
    interpolate: bool = False
    interpolate_fps: Optional[float] = None
    interpolate_backend: str = "minterpolate"   # or "rife"
    rife_multiplier: int = 2
    remove_dead_space: bool = False

    def any_enabled(self) -> bool:
        return self.stabilize or self.interpolate


def postprocess(src: str, dst: str, cfg: PostOps, *, quality: str = "high") -> str:
    """Apply the enabled smoothing operations to a rendered file.

    Runs stabilize, then interpolation, chaining through a temp file so each
    stage sees the previous stage's output. Returns *dst*.
    """
    if not cfg.any_enabled():
        return src

    current = src
    tmp_files: List[str] = []
    try:
        if cfg.stabilize:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            tmp_files.append(tmp)
            stabilize_video(current, tmp, smoothing=cfg.stabilization_smoothing,
                            quality=quality)
            current = tmp
        if cfg.interpolate:
            target_fps = cfg.interpolate_fps or (probe_fps(current) * 2.0)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            tmp_files.append(tmp)
            if cfg.interpolate_backend == "rife" and rife_available():
                interpolate_rife(current, tmp, multiplier=cfg.rife_multiplier)
            else:
                interpolate_fps(current, tmp, target_fps, quality=quality)
            current = tmp
        if current != dst:
            if os.path.exists(dst):
                os.remove(dst)
            os.replace(current, dst)
            if current in tmp_files:
                tmp_files.remove(current)
        return dst
    finally:
        for p in tmp_files:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Tiny CLI: ``python post_ops.py --filters`` / ``--silences FILE``."""
    import argparse
    ap = argparse.ArgumentParser(description="BeatSync post-processing ops")
    ap.add_argument("--filters", action="store_true", help="list relevant ffmpeg filters")
    ap.add_argument("--silences", metavar="FILE", help="print silence intervals")
    args = ap.parse_args(argv)

    if args.filters:
        for name in ("vidstabdetect", "vidstabtransform", "minterpolate", "silencedetect"):
            print(f"{name}: {'yes' if has_filter(name) else 'no'}")
        print(f"rife: {'yes' if rife_available() else 'no'}")
        return 0
    if args.silences:
        for s in detect_silences(args.silences):
            print(f"{s.start:.3f} - {s.end:.3f}  ({s.duration:.3f}s)")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
