#!/usr/bin/env python3
"""Beat / downbeat *refinement* with a bidirectional Transformer.

Model: ``livechord-music/livechord-beat-refiner`` (Apache-2.0). It is a
post-processing pass: it takes an initial beat grid from any tracker (here the
BeatTCN in ``tcn.py`` or librosa) plus the audio, and returns a corrected grid
that fixes phase drift and bar misalignment. The authors report beat F1 0.920
and downbeat F1 0.936 on a gold-quality holdout, which is what makes it useful
on live/acoustic Indic material where a grid tracker alone drifts.

The ``BeatRefiner`` definition is vendored from the model's ``model.py`` so the
state-dict keys match; the feature stack is vendored from ``features.py``. The
only intentional change is that audio is decoded with librosa (no soundfile
requirement) and the HF cache honours this repo's env overrides.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ._common import (
    REFINER_REPO_ID,
    SAMPLE_RATE,
    fetch_weight,
    load_mono_audio,
    require_torch,
)

# ── Frame-axis spec (must match the checkpoint exactly) ────────────────
HOP_LENGTH = 2048
N_CQT_BINS = 84
BINS_PER_OCTAVE = 12
FMIN_NOTE = "C1"

FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH      # ≈ 10.7666
N_AUDIO_CHANNELS = N_CQT_BINS + 12 + 1 + 1     # = 98
N_GRID_CHANNELS = 2
N_MODEL_CHANNELS = N_AUDIO_CHANNELS + N_GRID_CHANNELS  # = 100

MAX_AUDIO_SECONDS = 900.0   # 15 min cap, mirrors upstream
MODEL_VERSION = 1

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
FFN_DIM = 512
DROPOUT = 0.1
MAX_FRAMES = 9728

WEIGHT_FILENAMES = ("model.safetensors", "beat_refiner.pt")
LOCAL_DIR_ENV = "BEATSYNC_BEAT_REFINER_DIR"

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_DISTANCE_BEAT = 2
DEFAULT_MIN_DISTANCE_DOWNBEAT = 4


# ---------------------------------------------------------------------------
# Feature extraction (audio-only, 98 channels)
# ---------------------------------------------------------------------------

def time_to_frame(t: float) -> int:
    return int(round(float(t) * FRAMES_PER_SEC))


def frame_to_time(f: int) -> float:
    return float(f) / FRAMES_PER_SEC


def rasterize_times(times: List[float], n_frames: int) -> np.ndarray:
    grid = np.zeros(n_frames, dtype=np.float32)
    for t in times:
        f = time_to_frame(t)
        if 0 <= f < n_frames:
            grid[f] = 1.0
    return grid


def extract_features(audio_path: str = None, y: np.ndarray = None) -> Dict:
    """Return the 98-channel audio feature stack: (T, 98).

    ``y`` optionally supplies pre-decoded mono audio at 22050 Hz to avoid a
    second decode of the same file.
    """
    import librosa

    if y is None:
        y = load_mono_audio(audio_path, sr=SAMPLE_RATE)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("empty audio")
    max_samples = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)
    if y.size > max_samples:
        y = y[:max_samples]
    duration = len(y) / SAMPLE_RATE

    cqt_mag = np.abs(librosa.cqt(
        y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        n_bins=N_CQT_BINS, bins_per_octave=BINS_PER_OCTAVE,
        fmin=librosa.note_to_hz(FMIN_NOTE),
    )).astype(np.float32)
    log_cqt = librosa.amplitude_to_db(cqt_mag, ref=np.max).astype(np.float32) / 80.0
    log_cqt = np.clip(log_cqt, -1.0, 0.0)

    n_oct = N_CQT_BINS // BINS_PER_OCTAVE
    chroma = cqt_mag[:n_oct * BINS_PER_OCTAVE].reshape(n_oct, BINS_PER_OCTAVE, -1).sum(axis=0)
    chroma = (chroma / (chroma.sum(axis=0, keepdims=True) + 1e-8)).astype(np.float32)

    onset = librosa.onset.onset_strength(
        y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH).astype(np.float32)
    onset_max = float(onset.max()) if onset.size else 1.0
    if onset_max > 1e-6:
        onset = onset / onset_max

    rms = librosa.feature.rms(
        y=y, frame_length=HOP_LENGTH * 2, hop_length=HOP_LENGTH)[0].astype(np.float32)
    rms_max = float(rms.max()) if rms.size else 1.0
    if rms_max > 1e-6:
        rms = rms / rms_max

    T = min(log_cqt.shape[1], chroma.shape[1], onset.shape[0], rms.shape[0])
    features = np.empty((T, N_AUDIO_CHANNELS), dtype=np.float32)
    features[:, :N_CQT_BINS] = log_cqt[:, :T].T
    features[:, N_CQT_BINS:N_CQT_BINS + 12] = chroma[:, :T].T
    features[:, N_CQT_BINS + 12] = onset[:T]
    features[:, N_CQT_BINS + 13] = rms[:T]

    return {"features": features, "duration_sec": duration}


def build_initial_grid_channels(audio_features: np.ndarray,
                                beats: List[float],
                                downbeats: List[float]) -> np.ndarray:
    """Concat the 98 audio channels with beat/downbeat one-hot grids -> (T, 100)."""
    if audio_features.shape[1] != N_AUDIO_CHANNELS:
        raise ValueError(
            f"audio_features has {audio_features.shape[1]} channels, "
            f"expected {N_AUDIO_CHANNELS}"
        )
    T = audio_features.shape[0]
    out = np.empty((T, N_MODEL_CHANNELS), dtype=np.float32)
    out[:, :N_AUDIO_CHANNELS] = audio_features
    out[:, N_AUDIO_CHANNELS] = rasterize_times(list(beats), T)
    out[:, N_AUDIO_CHANNELS + 1] = rasterize_times(list(downbeats), T)
    return out


# ---------------------------------------------------------------------------
# Vendored architecture (livechord-music/livechord-beat-refiner model.py)
# ---------------------------------------------------------------------------

def _build_model_class():
    torch = require_torch()
    import torch.nn as nn

    def _build_sinusoidal_pos_enc(max_len: int, d_model: int) -> "torch.Tensor":
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    class BeatRefiner(nn.Module):
        def __init__(self, n_input_channels: int = N_MODEL_CHANNELS,
                     d_model: int = D_MODEL, n_heads: int = N_HEADS,
                     n_layers: int = N_LAYERS, ffn_dim: int = FFN_DIM,
                     dropout: float = DROPOUT, max_frames: int = MAX_FRAMES):
            super().__init__()
            self.d_model = d_model
            self.max_frames = max_frames
            self.input_proj = nn.Linear(n_input_channels, d_model)
            pe = _build_sinusoidal_pos_enc(max_frames, d_model)
            self.register_buffer("pos_enc", pe, persistent=False)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.encoder_norm = nn.LayerNorm(d_model)
            self.beat_head = nn.Linear(d_model, 1)
            self.downbeat_head = nn.Linear(d_model, 1)
            self.chord_boundary_head = nn.Linear(d_model, 1)
            for m in (self.input_proj, self.beat_head, self.downbeat_head,
                      self.chord_boundary_head):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        def forward(self, x, padding_mask=None):
            B, T, C = x.shape
            if T > self.max_frames:
                raise ValueError(f"Input has {T} frames; max_frames={self.max_frames}.")
            h = self.input_proj(x) + self.pos_enc[:T].unsqueeze(0)
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            h = self.encoder_norm(h)
            return {
                "beat_logits": self.beat_head(h).squeeze(-1),
                "downbeat_logits": self.downbeat_head(h).squeeze(-1),
                "chord_boundary_logits": self.chord_boundary_head(h).squeeze(-1),
            }

    return torch, BeatRefiner


# ---------------------------------------------------------------------------
# Loading + inference
# ---------------------------------------------------------------------------

_CACHE: Dict[str, object] = {}


def _load_state_dict(path: str):
    torch = require_torch()
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            return load_file(path)
        except Exception:
            return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        if ckpt.get("model_version") != MODEL_VERSION:
            return None
        return ckpt["model_state"]
    return ckpt


def load_model(device: str = "cpu"):
    """Load (and cache) the BeatRefiner. Returns ``(model, torch)``."""
    if device in _CACHE:
        return _CACHE[device]

    weight_path = None
    for fname in WEIGHT_FILENAMES:
        weight_path = fetch_weight(REFINER_REPO_ID, fname, LOCAL_DIR_ENV)
        if weight_path:
            break
    if weight_path is None:
        raise FileNotFoundError(
            "BeatRefiner weights not found. Provide them offline via "
            f"{LOCAL_DIR_ENV}=<dir> containing model.safetensors, or install "
            "huggingface_hub so they can be downloaded from "
            f"{REFINER_REPO_ID}."
        )

    state_dict = _load_state_dict(weight_path)
    if state_dict is None:
        raise RuntimeError(f"could not load beat-refiner weights from {weight_path}")

    torch, BeatRefiner = _build_model_class()
    model = BeatRefiner()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    _CACHE[device] = (model, torch)
    return _CACHE[device]


def peak_pick(probs: np.ndarray, threshold: float, min_distance: int) -> List[int]:
    """Local-maximum peak picker (identical to upstream ``_peak_pick``)."""
    probs = np.asarray(probs, dtype=float).reshape(-1)
    peaks: List[int] = []
    last = -10 ** 9
    for i in range(1, len(probs) - 1):
        if probs[i] < threshold:
            continue
        if probs[i] < probs[i - 1] or probs[i] <= probs[i + 1]:
            continue
        if i - last < min_distance:
            continue
        peaks.append(i)
        last = i
    return peaks


def is_available() -> bool:
    from ._common import torch_available

    if not torch_available():
        return False
    return any(
        fetch_weight(REFINER_REPO_ID, f, LOCAL_DIR_ENV) is not None
        for f in WEIGHT_FILENAMES
    )


def refine(audio_path: str, beats: List[float], downbeats: List[float],
           threshold: float = DEFAULT_THRESHOLD,
           min_distance_beat: int = DEFAULT_MIN_DISTANCE_BEAT,
           min_distance_downbeat: int = DEFAULT_MIN_DISTANCE_DOWNBEAT,
           device: str = "cpu", y: np.ndarray = None) -> Dict:
    """Refine ``beats``/``downbeats`` against the audio.

    Never raises across the boundary: on any failure the input grid is returned
    with ``applied=False`` and a short ``reason``. Callers should check
    ``applied`` before trusting the refined values.
    """
    beats = [float(b) for b in (beats or [])]
    downbeats = [float(d) for d in (downbeats or [])]
    out: Dict = {
        "refined_beats": list(beats),
        "refined_downbeats": list(downbeats),
        "applied": False,
        "reason": "",
        "n_beats_in": len(beats),
        "n_beats_out": len(beats),
        "n_downbeats_in": len(downbeats),
        "n_downbeats_out": len(downbeats),
    }
    if not beats:
        out["reason"] = "no-input-beats"
        return out

    try:
        model, torch = load_model(device)
    except Exception as exc:
        out["reason"] = f"no-model: {exc}"
        return out

    try:
        feat = extract_features(audio_path, y=y)
        full_input = build_initial_grid_channels(feat["features"], beats, downbeats)
    except Exception as exc:
        out["reason"] = f"audio-extract-failed: {type(exc).__name__}: {exc}"
        return out

    audio_seconds = float(feat.get("duration_sec") or 0.0)
    T = full_input.shape[0]
    if T > MAX_FRAMES:
        full_input = full_input[:MAX_FRAMES]
        T = MAX_FRAMES

    try:
        with torch.no_grad():
            x = torch.from_numpy(full_input).unsqueeze(0).to(device)
            pad_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
            res = model(x, padding_mask=pad_mask)
            beat_probs = torch.sigmoid(res["beat_logits"][0]).cpu().numpy()
            db_probs = torch.sigmoid(res["downbeat_logits"][0]).cpu().numpy()
    except Exception as exc:
        out["reason"] = f"inference-failed: {type(exc).__name__}: {exc}"
        return out

    beat_frames = peak_pick(beat_probs, threshold, min_distance_beat)
    db_frames = peak_pick(db_probs, threshold, min_distance_downbeat)

    out["refined_beats"] = [round(frame_to_time(f), 4) for f in beat_frames]
    out["refined_downbeats"] = [round(frame_to_time(f), 4) for f in db_frames]
    out["n_beats_out"] = len(out["refined_beats"])
    out["n_downbeats_out"] = len(out["refined_downbeats"])
    # Frames actually seen by the model. Callers must not adopt a refined grid
    # that only covers a prefix of a longer track (see MAX_FRAMES).
    out["covered_seconds"] = round(min(audio_seconds, T / FRAMES_PER_SEC), 4)
    out["input_seconds"] = round(audio_seconds, 4)
    out["applied"] = True
    out["reason"] = "ok"
    return out
