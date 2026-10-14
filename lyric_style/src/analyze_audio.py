"""Stage 3 — Audio analysis.

For each music file produce track_analysis.json (Section 8.2):
  bpm + confidence, beat times, downbeat/bar times, section boundaries with
  candidate "lift" (B) and "payoff" (C) points, loudness (LUFS) + true peak,
  and the top-N recommended 30 s windows.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np

from util import MissingAssetError, resolve, run


# --------------------------------------------------------------------------- #
# Loudness via ffmpeg ebur128 (reliable), pyloudnorm fallback
# --------------------------------------------------------------------------- #
def measure_loudness(path: str, ffmpeg: str) -> dict:
    res = run([ffmpeg, "-hide_banner", "-nostats", "-i", path,
               "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
              check=False)
    text = res.stderr or ""
    lufs = tpeak = lra = None
    # Summary block appears at the end.
    for m in re.finditer(r"I:\s*(-?\d+\.\d+)\s*LUFS", text):
        lufs = float(m.group(1))
    for m in re.finditer(r"Peak:\s*(-?\d+\.\d+)\s*dBFS", text):
        tpeak = float(m.group(1))
    for m in re.finditer(r"LRA:\s*(-?\d+\.\d+)\s*LU", text):
        lra = float(m.group(1))
    if lufs is None:
        try:  # pragma: no cover - optional dependency
            import pyloudnorm as pyln
            import soundfile as sf
            data, rate = sf.read(path)
            meter = pyln.Meter(rate)
            lufs = float(meter.integrated_loudness(data))
            tpeak = float(20 * math.log10(max(1e-9, np.abs(data).max())))
        except Exception:
            lufs = tpeak = None
    # ebur128 Peak reports sample peak; true peak can sit ~0.4 dB higher.
    # Leave headroom so the encoder ceiling holds after AAC.
    true_peak = (tpeak + 0.4) if tpeak is not None else None
    return {"lufs": lufs, "true_peak_dbfs": true_peak, "sample_peak_dbfs": tpeak, "lra_lu": lra}


# --------------------------------------------------------------------------- #
# Beats, downbeats, sections
# --------------------------------------------------------------------------- #
def analyze(path: str, cfg: dict, ffmpeg: str) -> dict:
    import librosa

    sr = cfg["audio_analysis"]["sr"]
    hop = cfg["audio_analysis"]["hop_length"]
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise MissingAssetError(f"{path}: no audio decoded")
    duration = float(len(y)) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop,
                                                 units="frames")
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop).tolist()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    downbeats, phase = estimate_downbeats(beats, onset_env, beat_frames)

    sections = estimate_sections(y, sr, hop)
    lift, payoff = pick_lift_payoff(sections, y, sr, hop)

    windows = recommend_windows(path, duration, bpm, downbeats, onset_env,
                                beat_frames, sr, hop,
                                cfg["audio_analysis"]["window_seconds"],
                                cfg["audio_analysis"]["recommend_top_windows"])

    loud = measure_loudness(path, ffmpeg)

    return {
        "path": str(Path(path).resolve()),
        "duration": duration,
        "bpm": round(bpm, 3),
        "bpm_confidence": round(beat_confidence(onset_env, beat_frames), 3),
        "beats": [round(b, 4) for b in beats],
        "downbeats": [round(b, 4) for b in downbeats],
        "downbeat_phase": phase,
        "meter": 4,
        "sections": sections,
        "lift_candidates": lift,
        "payoff_candidates": payoff,
        "loudness": loud,
        "recommended_windows": windows,
    }


def beat_confidence(onset_env: np.ndarray, beat_frames: np.ndarray) -> float:
    if beat_frames.size < 4:
        return 0.0
    vals = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    if vals.std() < 1e-9:
        return 0.0
    # Higher mean relative to spread => steadier, more confident beat.
    return float(min(1.0, vals.mean() / (vals.std() + vals.mean() + 1e-9)))


def estimate_downbeats(beats: list[float], onset_env: np.ndarray,
                       beat_frames: np.ndarray, meter: int = 4):
    """Pick the phase (0..meter-1) whose beats carry the most onset energy."""
    if len(beats) < meter:
        return list(beats), 0
    strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    best_phase, best_score = 0, -1.0
    for phase in range(meter):
        idx = np.arange(phase, len(strengths), meter)
        if idx.size == 0:
            continue
        score = float(strengths[idx].mean())
        if score > best_score:
            best_phase, best_score = phase, score
    return [beats[i] for i in range(best_phase, len(beats), meter)], best_phase


def estimate_sections(y: np.ndarray, sr: int, hop: int) -> list[dict]:
    """Novelty-based boundaries from a smoothed RMS + spectral-change curve."""
    rms = librosa_feature_rms(y, hop)
    n = len(rms)
    if n < 8:
        return []
    # Smooth, then find large positive jumps (energy lifts).
    win = max(3, n // 40)
    kernel = np.ones(win) / win
    smooth = np.convolve(rms, kernel, mode="same")
    diff = np.diff(smooth, prepend=smooth[0])
    thr = diff.std() * 1.5 + 1e-9
    peaks = np.where(diff > thr)[0]
    times = []
    for p in peaks:
        t = p * hop / sr
        if not times or t - times[-1] > 1.0:
            times.append(round(float(t), 3))
    out = [{"time": 0.0, "kind": "start"}]
    for t in times:
        out.append({"time": t, "kind": "boundary"})
    out.append({"time": round(n * hop / sr, 3), "kind": "end"})
    return out


def librosa_feature_rms(y: np.ndarray, hop: int) -> np.ndarray:
    import librosa
    S = np.abs(librosa.stft(y, hop_length=hop))
    rms = librosa.feature.rms(S=S)[0]
    return rms


def pick_lift_payoff(sections: list[dict], y: np.ndarray, sr: int, hop: int):
    """Choose the 3 best 'lift' (B start) and 'payoff' (C start) candidates."""
    if not sections:
        return [], []
    rms = librosa_feature_rms(y, hop)
    total = len(rms)
    bounds = [s for s in sections if s["kind"] == "boundary"]
    if not bounds:
        mid = round(total * hop / sr * 0.5, 3)
        return [mid], [round(mid * 1.2, 3)]
    scored = []
    for b in bounds:
        i = int(b["time"] * sr / hop)
        if i <= 0 or i >= total:
            continue
        before = rms[max(0, i - 40):i].mean() if i > 0 else 0.0
        after = rms[i:i + 40].mean() if i + 40 < total else rms[i:].mean()
        scored.append((b["time"], float(after - before)))
    scored.sort(key=lambda x: x[1], reverse=True)
    lift = [round(t, 3) for t, _ in scored[:3]]
    later = sorted([s for s in scored if s[0] > (lift[0] if lift else 0)], key=lambda x: x[0])
    payoff = [round(t, 3) for t, _ in later[:3]]
    return lift, payoff


def recommend_windows(path, duration, bpm, downbeats, onset_env, beat_frames,
                      sr, hop, win_s, top_n) -> list[dict]:
    """Rank 30 s windows: start on a downbeat, stable level, phrase-aligned end."""
    if duration <= win_s:
        return [{"start": 0.0, "end": round(duration, 3), "score": 1.0}]
    db = [d for d in downbeats if d + win_s <= duration + 1e-6]
    if not db:
        step = win_s / 2
        db = [i * step for i in range(int((duration - win_s) / step) + 1)]
    on_env = onset_env / (onset_env.max() + 1e-9)
    cands = []
    for start in db:
        i0 = int(start * sr / hop)
        i1 = int((start + win_s) * sr / hop)
        seg = on_env[i0:i1]
        if seg.size == 0:
            continue
        density = float(seg.mean())
        # Reward a strong onset right at the start (vocal phrase entry).
        entry = float(seg[:max(1, len(seg) // 20)].mean())
        # Reward ending near a downbeat.
        end_t = start + win_s
        align = min(abs(end_t - d) for d in downbeats) if downbeats else 0.0
        align_score = max(0.0, 1.0 - align / (60.0 / max(bpm, 1e-6)))
        score = 0.4 * density + 0.4 * entry + 0.2 * align_score
        cands.append({"start": round(start, 3), "end": round(end_t, 3),
                      "score": round(score, 4)})
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands[:top_n]


def main(argv: list[str] | None = None) -> int:
    import argparse
    from util import load_config, require_ffmpeg, write_json
    ap = argparse.ArgumentParser(description="Stage 3: analyse music tracks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--music", nargs="*", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ffmpeg, _ = require_ffmpeg()
    mdir = resolve(cfg, "music_dir")
    paths = args.music or [
        str(p) for p in sorted(mdir.glob("*"))
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
    ]
    if not paths:
        print(f"No music found in {mdir}. Add tracks or pass --music.")
        return 2
    for path in paths:
        info = analyze(path, cfg, ffmpeg)
        out = Path(path).with_suffix(".track_analysis.json")
        write_json(out, info)
        print(f"✅ {Path(path).name}: {info['bpm']} BPM, "
              f"{len(info['downbeats'])} downbeats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
