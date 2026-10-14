#!/usr/bin/env python3
"""Neural beat detection with a Temporal Convolutional Network.

Architecture and weights: ``finnvoorhees/beat-detection-tcn`` (MIT) — a
~269K-parameter TCN (All-In-One / BeatNet lineage) that reads a log-mel
spectrogram at 22050 Hz / 100 fps and emits a frame-level beat activation.
It is far more robust on dense percussive material (dhol/tabla rolls, off-beat
syncopation) than ``librosa.beat.beat_track``, which is why it is the preferred
front-end for Indic tracks.

The ``BeatTCN`` / ``TCNBlock`` definitions below are vendored verbatim from the
model's ``train_beat_detector.py`` so the state-dict keys match exactly; do not
"clean them up" without re-checking against upstream. Only the feature
extraction differs from upstream: librosa computes the mel spectrogram
(verified to match ``torchaudio.transforms.MelSpectrogram`` to float32
precision), so the repo needs no torchaudio dependency.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from ._common import TCN_REPO_ID, load_mono_audio, require_torch, fetch_weight

WEIGHT_FILENAME = "model.pt"
LOCAL_DIR_ENV = "BEATSYNC_BEAT_MODEL_DIR"

# Default peak-picking parameters. ``threshold`` is the activation cutoff;
# ``min_interval`` is the shortest gap between accepted beats (seconds).
# Upstream defaults are 0.3 / 0.2; we expose them via the environment so an
# Indic track with fast rolls can be tuned without a code change.
DEFAULT_THRESHOLD = float(os.environ.get("BEATSYNC_TCN_THRESHOLD", "0.3"))
DEFAULT_MIN_INTERVAL = float(os.environ.get("BEATSYNC_TCN_MIN_INTERVAL", "0.2"))


def _build():
    """Import torch/nn and define the vendored architecture. Returns
    ``(torch, nn, Config, TCNBlock, BeatTCN)``."""
    torch = require_torch()
    import torch.nn as nn


    class Config:
        """Mirror of the upstream training config (train_beat_detector.py).

        Every field is set from the checkpoint's ``config`` dict at load time,
        so the values here are only fallbacks for a config-less checkpoint.
        """

        sample_rate: int = 22050
        n_fft: int = 2048
        hop_length: int = 220
        n_mels: int = 81
        fmin: float = 30.0
        fmax: float = 11000.0

        tcn_channels: int = 48
        tcn_layers: int = 11
        tcn_kernel_size: int = 5
        dropout: float = 0.15

    class TCNBlock(nn.Module):
        """Dilated causal conv block with residual connection (All-In-One)."""

        def __init__(self, channels: int, kernel_size: int, dilation1: int,
                     dilation2: int, dropout: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv1d(
                channels, channels, kernel_size,
                padding=(kernel_size - 1) * dilation1 // 2,
                dilation=dilation1,
            )
            self.bn1 = nn.BatchNorm1d(channels)
            self.drop1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(
                channels, channels, kernel_size,
                padding=(kernel_size - 1) * dilation2 // 2,
                dilation=dilation2,
            )
            self.bn2 = nn.BatchNorm1d(channels)
            self.drop2 = nn.Dropout(dropout)

            self.elu = nn.ELU()

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.elu(out)
            out = self.drop1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.elu(out)
            out = self.drop2(out)
            return out + residual

    class BeatTCN(nn.Module):
        """Mel frontend -> dilated TCN stack -> 1-D beat activation head."""

        def __init__(self, config: Config):
            super().__init__()
            self.config = config

            self.frontend = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.MaxPool2d((3, 1)),

                nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.MaxPool2d((3, 1)),

                nn.Conv2d(16, 16, kernel_size=(1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
            )

            freq_dim = config.n_mels // 9
            frontend_out = 16 * freq_dim

            self.projection = nn.Sequential(
                nn.Linear(frontend_out, config.tcn_channels),
                nn.ELU(),
            )

            self.tcn_blocks = nn.ModuleList()
            for i in range(config.tcn_layers):
                dilation1 = 2 ** i
                dilation2 = 2 ** (i + 1)
                self.tcn_blocks.append(
                    TCNBlock(
                        config.tcn_channels,
                        config.tcn_kernel_size,
                        dilation1,
                        dilation2,
                        config.dropout,
                    )
                )

            self.output_head = nn.Sequential(
                nn.Conv1d(config.tcn_channels, config.tcn_channels, 1),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(config.tcn_channels, 1, 1),
            )

        def forward(self, mel):
            """``mel``: (batch, n_mels, n_frames) -> (batch, n_frames) logits."""
            x = mel.unsqueeze(1)
            x = self.frontend(x)
            b, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
            x = self.projection(x)
            x = x.permute(0, 2, 1)
            for block in self.tcn_blocks:
                x = block(x)
            return self.output_head(x).squeeze(1)

    return torch, nn, Config, TCNBlock, BeatTCN


_CACHE: Dict[str, object] = {}


def _log_mel(y: np.ndarray, cfg) -> np.ndarray:
    """Log-mel spectrogram matching torchaudio's MelSpectrogram(power=2.0)."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.n_fft, window="hann", center=True, pad_mode="reflect",
        power=2.0, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
        htk=True, norm=None,
    )
    return np.log1p(mel)


def peak_pick(activations: np.ndarray, threshold: float,
              min_interval: float, fps: float) -> np.ndarray:
    """Greedy peak picker identical to the upstream ``BeatDetector.detect``.

    Walks the activation curve; whenever a frame clears ``threshold`` it takes
    the local maximum within the next ``min_interval`` window as a beat and
    skips ahead. Returns beat times in seconds.
    """
    activations = np.asarray(activations, dtype=float).reshape(-1)
    if activations.size == 0:
        return np.zeros(0, dtype=float)

    min_frames = max(1, int(min_interval * fps))
    beats = []
    i = 0
    n = activations.size
    while i < n:
        if activations[i] >= threshold:
            window_end = min(i + min_frames, n)
            peak_idx = i + int(np.argmax(activations[i:window_end]))
            beats.append(peak_idx / fps)
            i = peak_idx + min_frames
        else:
            i += 1
    return np.asarray(beats, dtype=float)


def is_available() -> bool:
    """True when torch is importable and the checkpoint can be resolved."""
    from ._common import torch_available

    if not torch_available():
        return False
    return fetch_weight(TCN_REPO_ID, WEIGHT_FILENAME, LOCAL_DIR_ENV) is not None


def load_detector(device: str = "cpu"):
    """Load (and cache) the BeatTCN model. Returns ``(model, config, torch)``."""
    if device in _CACHE:
        return _CACHE[device]

    torch, _nn, Config, _TCNBlock, BeatTCN = _build()

    weight_path = fetch_weight(TCN_REPO_ID, WEIGHT_FILENAME, LOCAL_DIR_ENV)
    if weight_path is None:
        raise FileNotFoundError(
            "BeatTCN weights not found. Provide them offline via "
            f"{LOCAL_DIR_ENV}=<dir> containing {WEIGHT_FILENAME}, or install "
            "huggingface_hub so they can be downloaded from "
            f"{TCN_REPO_ID}."
        )

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    config = Config()
    for key, value in (checkpoint.get("config") or {}).items():
        setattr(config, key, value)

    model = BeatTCN(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    _CACHE[device] = (model, config, torch)
    return _CACHE[device]


def detect_beats(audio_path: str = None, threshold: float = None,
                 min_interval: float = None, device: str = "cpu",
                 y: np.ndarray = None) -> np.ndarray:
    """Detect beat times (seconds) in ``audio_path`` with the BeatTCN.

    Args:
        audio_path: path to the audio file (ignored when ``y`` is given).
        threshold: activation cutoff (default ``BEATSYNC_TCN_THRESHOLD``/0.3).
        min_interval: minimum seconds between beats
            (default ``BEATSYNC_TCN_MIN_INTERVAL``/0.2).
        device: torch device string.
        y: pre-decoded mono audio at ``cfg.sample_rate``; pass this to reuse an
            already-loaded buffer instead of decoding the file twice.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if min_interval is None:
        min_interval = DEFAULT_MIN_INTERVAL

    model, cfg, torch = load_detector(device)
    if y is None:
        y = load_mono_audio(audio_path, sr=cfg.sample_rate)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return np.zeros(0, dtype=float)

    mel = _log_mel(y, cfg)
    with torch.no_grad():
        mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)
        logits = model(mel_t)
        activations = torch.sigmoid(logits).squeeze().cpu().numpy()

    fps = cfg.sample_rate / cfg.hop_length
    return peak_pick(activations, threshold, min_interval, fps)


def estimate_tempo(beat_times: np.ndarray) -> float:
    """BPM from the median beat interval (0.0 when there are too few beats)."""
    beat_times = np.asarray(beat_times, dtype=float)
    if beat_times.size < 2:
        return 0.0
    intervals = np.diff(np.sort(beat_times))
    intervals = intervals[intervals > 1e-6]
    if intervals.size == 0:
        return 0.0
    return float(60.0 / np.median(intervals))
