#!/usr/bin/env python3
"""Stage 7: snap planned cuts onto real transient onsets.

Beat tracking yields a *regular* grid, but the actual musical accent (kick,
snare, vocal stab) can sit tens of milliseconds off that grid. Rendering a cut
on the grid value instead of the accent is the single biggest reason an edit
feels "almost synced" rather than locked. This stage nudges each cut onto the
nearest detected onset within a small tolerance, so every shot change lands on
an audible transient.
"""

from typing import Dict, List

import librosa
import numpy as np

from . import AutoWaveConfig, _unique_sorted


def detect_onsets(y_percussive: np.ndarray, sr: int, cfg: AutoWaveConfig) -> np.ndarray:
    """Return onset times (seconds) from the percussive signal."""
    if y_percussive is None or len(y_percussive) == 0:
        return np.zeros(0, dtype=float)
    try:
        frames = librosa.onset.onset_detect(
            y=y_percussive,
            sr=sr,
            hop_length=cfg.hop_length,
            backtrack=True,
            units="frames",
        )
        times = librosa.frames_to_time(np.asarray(frames, dtype=int), sr=sr,
                                       hop_length=cfg.hop_length)
    except Exception:
        return np.zeros(0, dtype=float)
    times = np.asarray(times, dtype=float)
    return times[np.isfinite(times) & (times >= 0.0)]


def snap_cuts_to_onsets(
    selected: np.ndarray,
    onset_times: np.ndarray,
    audio_duration: float,
    cfg: AutoWaveConfig,
    strength: float = 1.0,
) -> np.ndarray:
    """Nudge each cut to the nearest onset within a tolerance window.

    Args:
        selected: planned cut times (seconds), excluding the locked 0.0 start.
        onset_times: detected transient times.
        audio_duration: total audio length (cuts are clamped inside).
        cfg: provides the base tolerance and the no-flicker gap.
        strength: 0 disables snapping, 1 uses the full tolerance.

    Returns:
        A sorted, de-duplicated cut array with the same first element order.
    """
    arr = np.asarray(selected, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr > 0.0) & (arr < audio_duration)]
    if arr.size == 0 or onset_times.size == 0:
        return arr

    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return np.sort(arr)

    tolerance = float(cfg.onset_snap_tolerance) * strength
    if tolerance <= 0.0:
        return np.sort(arr)

    onsets = np.sort(np.asarray(onset_times, dtype=float))
    snapped: List[float] = []
    for t in arr:
        # index of the closest onset (searchsorted on the sorted onset grid)
        pos = int(np.searchsorted(onsets, t))
        neighbours = onsets[max(0, pos - 1): pos + 1]
        if neighbours.size == 0:
            snapped.append(float(t))
            continue
        nearest = float(neighbours[np.argmin(np.abs(neighbours - t))])
        # Only move when the onset is genuinely closer to the musical event and
        # the shift is small; otherwise keep the grid value.
        if abs(nearest - t) <= tolerance:
            snapped.append(nearest)
        else:
            snapped.append(float(t))

    # Re-apply the no-flicker gap so snapping cannot create two cuts that are
    # too close together.
    return _unique_sorted(np.asarray(snapped, dtype=float),
                          max(cfg.peak_energy_min_interval, 0.05))


def summarize_snap(before: np.ndarray, after: np.ndarray,
                   onset_times: np.ndarray) -> Dict:
    """Report how far cuts moved and how close they now sit to an onset."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    moved = float(np.mean(np.abs(after - before))) if before.size == after.size and before.size else 0.0
    if onset_times.size and after.size:
        nearest = [float(np.min(np.abs(onset_times - t))) for t in after]
        mean_dist = float(np.mean(nearest))
        on_onset = float(np.mean([d <= 0.03 for d in nearest]))
    else:
        mean_dist = 0.0
        on_onset = 0.0
    return {
        "cuts_before": int(before.size),
        "cuts_after": int(after.size),
        "mean_shift_seconds": moved,
        "mean_onset_distance": mean_dist,
        "fraction_on_onset": on_onset,
    }
