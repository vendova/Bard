#!/usr/bin/env python3
"""Shared helpers for the optional neural beat-detection backends.

Every backend in this package is *optional*: the repo must keep working on a
plain CPU box with no PyTorch and no network. The helpers here give each
backend the same three guarantees:

* torch is imported lazily and its absence is reported, never raised;
* Hugging Face weights are cached under ``BEATSYNC_BEAT_MODEL_CACHE`` and can
  be pointed at a local directory (``BEATSYNC_BEAT_MODEL_DIR``) for offline
  use or tests;
* audio is always decoded to mono float32 at 22050 Hz with librosa, so the
  inference path never depends on torchaudio.
"""

from __future__ import annotations

import os
from typing import Optional

# All models in this package were trained at 22050 Hz.
SAMPLE_RATE = 22050

# Repos on the Hugging Face Hub.
TCN_REPO_ID = "finnvoorhees/beat-detection-tcn"
REFINER_REPO_ID = "livechord-music/livechord-beat-refiner"


def torch_available() -> bool:
    """True when PyTorch can be imported (the models are pure PyTorch)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def require_torch():
    """Import torch or raise a clear, actionable error."""
    try:
        import torch
        return torch
    except Exception as exc:  # pragma: no cover - exercised without torch
        raise RuntimeError(
            "Neural beat detection needs PyTorch. Install the optional extra:\n"
            "    pip install -r requirements-beats.txt\n"
            "(CPU-only wheels: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc


def model_cache_dir() -> str:
    """Directory where Hugging Face weights are cached.

    Defaults to ``<state dir>/beat_models`` when ``BEATSYNC_STATE_DIR`` is set
    (so a Render persistent disk keeps the weights across restarts), else
    ``~/.cache/beatsync_beat_models``.
    """
    env = os.environ.get("BEATSYNC_BEAT_MODEL_CACHE")
    if env:
        return os.path.expanduser(env)
    state = os.environ.get("BEATSYNC_STATE_DIR")
    if state:
        return os.path.join(os.path.expanduser(state), "beat_models")
    return os.path.expanduser("~/.cache/beatsync_beat_models")


def resolve_local_dir(env_var: str) -> Optional[str]:
    """Return a directory override from the environment, if it exists."""
    value = os.environ.get(env_var)
    if not value:
        return None
    path = os.path.expanduser(value)
    return path if os.path.isdir(path) else None


def fetch_weight(repo_id: str, filename: str, local_dir_env: str) -> Optional[str]:
    """Resolve a weight file, preferring a local override over the Hub.

    Resolution order:
      1. ``<local_dir_env>/<filename>`` (offline / tests / pre-downloaded);
      2. Hugging Face Hub download into ``model_cache_dir()``.

    Returns the path, or ``None`` when the file cannot be found and the Hub is
    unreachable. Never raises.
    """
    local_dir = resolve_local_dir(local_dir_env)
    if local_dir:
        candidate = os.path.join(local_dir, filename)
        if os.path.isfile(candidate):
            return candidate

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=model_cache_dir(),
        )
    except Exception:
        return None


def load_mono_audio(audio_path: str, sr: int = SAMPLE_RATE):
    """Decode ``audio_path`` to mono float32 at ``sr`` (librosa/FFmpeg)."""
    import librosa

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return y
