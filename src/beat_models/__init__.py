#!/usr/bin/env python3
"""Optional neural beat-detection backends for BeatSync-Engine.

Two independent, off-by-default models:

* :mod:`beat_models.tcn` — BeatTCN (``finnvoorhees/beat-detection-tcn``):
  a lightweight Temporal Convolutional Network that produces a frame-accurate
  beat grid from a log-mel spectrogram. Preferred over ``librosa.beat.beat_track``
  on dense percussive material (dhol/tabla rolls, off-beat syncopation).
* :mod:`beat_models.refiner` — BeatRefiner
  (``livechord-music/livechord-beat-refiner``): a bidirectional Transformer that
  refines the TCN/librosa grid and also emits a downbeat (bar) grid, which is
  what keeps bar/phrase anchors honest on live acoustic tracks.

Everything here is optional and lazy: importing this package does **not** import
torch. Without the extra in ``requirements-beats.txt`` (or without cached
weights) the functions report unavailability and the caller falls back to the
librosa pipeline.

The single entry point used by the render pipeline is
:func:`detect_beat_grid`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import refiner, tcn
from ._common import SAMPLE_RATE, torch_available

__all__ = [
    "detect_beat_grid",
    "is_available",
    "availability",
    "SAMPLE_RATE",
    "tcn",
    "refiner",
]


def is_available() -> bool:
    """True when the BeatTCN (the essential backend) can run."""
    return tcn.is_available()


def availability() -> Dict[str, bool]:
    """Report which backends are usable, without loading any weights."""
    return {
        "torch": torch_available(),
        "tcn": tcn.is_available(),
        "refiner": refiner.is_available(),
    }


def detect_beat_grid(
    audio_path: str,
    use_tcn: bool = True,
    use_refiner: bool = True,
    threshold: Optional[float] = None,
    min_interval: Optional[float] = None,
    device: str = "cpu",
    y: Optional[np.ndarray] = None,
) -> Dict:
    """Detect a beat grid with the neural backends.

    Returns a dict:
      beats        np.ndarray  beat times in seconds (empty on failure)
      tempo        float       BPM from the beat intervals (0.0 if unknown)
      downbeats    np.ndarray  bar-start times (every 4th beat when the
                               refiner is unavailable)
      backend      str         "tcn+refiner" | "tcn" | "unavailable"
      refined      bool        whether the Transformer refiner ran
      notes        str         human-readable status / failure reason

    This never raises: on any failure it returns ``backend="unavailable"`` with
    empty beats so the caller can fall back to librosa.
    """
    result = {
        "beats": np.zeros(0, dtype=float),
        "tempo": 0.0,
        "downbeats": np.zeros(0, dtype=float),
        "backend": "unavailable",
        "refined": False,
        "notes": "",
    }

    if not use_tcn:
        result["notes"] = "neural beat detection disabled"
        return result
    if not torch_available():
        result["notes"] = "PyTorch not installed (see requirements-beats.txt)"
        return result

    try:
        beats = tcn.detect_beats(audio_path, threshold=threshold,
                                 min_interval=min_interval, device=device, y=y)
    except Exception as exc:
        result["notes"] = f"TCN failed: {type(exc).__name__}: {exc}"
        return result

    beats = np.asarray(beats, dtype=float)
    if beats.size < 2:
        result["notes"] = "TCN produced too few beats"
        return result

    result["beats"] = beats
    result["tempo"] = tcn.estimate_tempo(beats)
    result["backend"] = "tcn"
    # Bar heuristic fallback: assume 4/4 and mark every 4th beat as a downbeat.
    downbeats = beats[::4]
    result["downbeats"] = np.asarray(downbeats, dtype=float)

    if use_refiner and refiner.is_available():
        try:
            out = refiner.refine(
                audio_path, beats.tolist(), result["downbeats"].tolist(),
                device=device, y=y,
            )
        except Exception as exc:
            result["notes"] = f"refiner failed: {type(exc).__name__}: {exc}"
            out = None
        if out is not None and out.get("applied"):
            refined = np.asarray(out["refined_beats"], dtype=float)
            # Reject a refined grid that only covers a prefix of the track
            # (the refiner caps its input at MAX_FRAMES) — the TCN grid is
            # full-length and preferable to a truncated refiner grid.
            covered = float(out.get("covered_seconds") or 0.0)
            input_end = float(out.get("input_seconds") or 0.0)
            truncated = covered > 0.0 and input_end > covered + 1.0
            if refined.size >= 2 and not truncated:
                result["beats"] = refined
                result["tempo"] = tcn.estimate_tempo(refined)
                ref_db = np.asarray(out["refined_downbeats"], dtype=float)
                if ref_db.size:
                    result["downbeats"] = ref_db
                result["backend"] = "tcn+refiner"
                result["refined"] = True
            elif truncated:
                result["notes"] = (
                    f"refiner covers only {covered:.0f}s of {input_end:.0f}s; "
                    "kept full-length TCN grid")
        elif out is not None:
            result["notes"] = f"refiner skipped: {out.get('reason', 'unknown')}"

    return result
