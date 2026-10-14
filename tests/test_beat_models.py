"""Tests for the optional neural beat-detection backends (src/beat_models).

These tests never touch the network: they build random-weight checkpoints in
the exact formats the real models ship in (a wrapped ``.pt`` for the TCN, a
``safetensors`` file for the refiner) and point the ``BEATSYNC_BEAT_*_DIR``
overrides at them. That exercises the real load + forward + peak-pick path
without a Hub download.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from media_paths import resolve_ffmpeg

torch = pytest.importorskip("torch", reason="neural beat tests need torch")
pytest.importorskip("safetensors", reason="refiner checkpoint is safetensors")

import beat_models
from beat_models import refiner, tcn

FFMPEG, _ = resolve_ffmpeg()
requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and Path(FFMPEG).exists()), reason="ffmpeg not available"
)


def _click_track(path, duration=8.0, bpm=120):
    interval = 60.0 / bpm
    filt = (f"aevalsrc=0.7*sin(2*PI*90*t)*exp(-35*mod(t\\,{interval}))"
            f":s=22050:d={duration}")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", filt,
         "-c:a", "pcm_s16le", str(path)], check=True)


@pytest.fixture
def tcn_checkpoint(tmp_path, monkeypatch):
    """Write a random-weight BeatTCN checkpoint and point the loader at it."""
    _, _, Config, _, BeatTCN = tcn._build()
    cfg = Config()
    model = BeatTCN(cfg)
    keys = ["sample_rate", "n_fft", "hop_length", "n_mels", "fmin", "fmax",
            "tcn_channels", "tcn_layers", "tcn_kernel_size", "dropout"]
    torch.save(
        {"config": {k: getattr(cfg, k) for k in keys},
         "model_state_dict": model.state_dict()},
        tmp_path / "model.pt",
    )
    monkeypatch.setenv("BEATSYNC_BEAT_MODEL_DIR", str(tmp_path))
    tcn._CACHE.clear()
    yield tmp_path
    tcn._CACHE.clear()


@pytest.fixture
def refiner_checkpoint(tmp_path, monkeypatch):
    """Write a random-weight BeatRefiner safetensors and point the loader at it."""
    from safetensors.torch import save_file

    _, BeatRefiner = refiner._build_model_class()
    model = BeatRefiner()
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(tmp_path / "model.safetensors"))
    monkeypatch.setenv("BEATSYNC_BEAT_REFINER_DIR", str(tmp_path))
    refiner._CACHE.clear()
    yield tmp_path
    refiner._CACHE.clear()


def test_peak_pick_respects_threshold_and_interval():
    fps = 100.0
    activations = np.zeros(300, dtype=float)
    activations[10] = 0.9    # beat
    activations[12] = 0.95   # too close -> swallowed by the 0.2s window
    activations[100] = 0.1   # below threshold -> ignored
    activations[200] = 0.8   # beat

    beats = tcn.peak_pick(activations, threshold=0.3, min_interval=0.2, fps=fps)

    assert len(beats) == 2
    assert np.isclose(beats[0], 0.12)   # local max inside the window
    assert np.isclose(beats[1], 2.00)


def test_estimate_tempo_from_beats():
    beats = np.arange(0.0, 4.0, 0.5)  # 120 BPM
    assert np.isclose(tcn.estimate_tempo(beats), 120.0, atol=1.0)
    assert tcn.estimate_tempo(np.array([1.0])) == 0.0


def test_refiner_peak_pick_is_local_maximum():
    probs = np.zeros(50, dtype=float)
    probs[10] = 0.9
    probs[11] = 0.95
    probs[30] = 0.8
    peaks = refiner.peak_pick(probs, threshold=0.5, min_distance=2)
    assert peaks == [11, 30]


def test_refiner_feature_channels_and_grid(tcn_checkpoint):
    # Shape contract only: the real CQT is expensive, so assert the channel
    # layout build_initial_grid_channels enforces.
    audio = np.zeros((20, refiner.N_AUDIO_CHANNELS), dtype=np.float32)
    grid = refiner.build_initial_grid_channels(audio, [0.0, 0.5], [0.0])
    assert grid.shape == (20, refiner.N_MODEL_CHANNELS)
    assert grid[:, refiner.N_AUDIO_CHANNELS].sum() == 2  # two beats rasterized
    assert grid[:, refiner.N_AUDIO_CHANNELS + 1].sum() == 1

    with pytest.raises(ValueError):
        refiner.build_initial_grid_channels(np.zeros((5, 3), dtype=np.float32),
                                            [0.0], [0.0])


@requires_ffmpeg
def test_tcn_detects_beats_from_checkpoint(tmp_path, tcn_checkpoint):
    audio = tmp_path / "click.wav"
    _click_track(audio)
    beats = tcn.detect_beats(str(audio))
    assert beats.ndim == 1
    assert np.all(np.diff(beats) > 0)


@requires_ffmpeg
def test_detect_beat_grid_tcn_plus_refiner(tmp_path, tcn_checkpoint, refiner_checkpoint):
    audio = tmp_path / "click.wav"
    _click_track(audio)
    result = beat_models.detect_beat_grid(str(audio))
    # A random TCN still yields a monotonic grid. The refiner may replace the
    # grid only when its peak-pick finds beats; either backend is acceptable.
    assert result["backend"] in ("tcn", "tcn+refiner")
    assert result["beats"].size >= 2
    assert np.all(np.diff(result["beats"]) > 0)
    assert result["downbeats"].size >= 1


@requires_ffmpeg
def test_refiner_refine_runs_end_to_end(tmp_path, refiner_checkpoint):
    audio = tmp_path / "click.wav"
    _click_track(audio)
    beats = np.arange(0.0, 8.0, 0.5).tolist()
    out = refiner.refine(str(audio), beats, beats[::4])
    # Real load + CQT features + Transformer forward + peak-pick path.
    assert out["applied"] is True
    assert out["reason"] == "ok"
    assert out["n_beats_in"] == len(beats)


@requires_ffmpeg
def test_analyze_beats_auto_uses_neural_grid(tmp_path, tcn_checkpoint, refiner_checkpoint, monkeypatch):
    from auto_mode import analyze_beats_auto

    audio = tmp_path / "click.wav"
    _click_track(audio)

    _, info = analyze_beats_auto(
        str(audio), video_files=[], enable_video_analysis=False,
        enable_qwen_semantics=False, min_shot_seconds=2.0,
        enable_neural_beats=True,
    )
    neural = info["neural_beats"]
    assert neural["enabled"] is True
    assert neural["used"] is True
    assert neural["backend"] in ("tcn", "tcn+refiner")
    # Bar anchors are derived from the neural downbeat grid.
    assert info["rhythm_data"]["is_bar_anchor"].any()


def test_detect_beat_grid_falls_back_without_torch(monkeypatch):
    monkeypatch.setattr(beat_models, "torch_available", lambda: False)
    result = beat_models.detect_beat_grid("ignored.wav")
    assert result["backend"] == "unavailable"
    assert result["beats"].size == 0
    assert "PyTorch" in result["notes"]


def test_detect_beat_grid_keeps_full_tcn_grid_when_refiner_truncates(monkeypatch):
    # A 20-min track: the refiner caps its input at MAX_FRAMES (~15 min), so a
    # refined grid would only cover a prefix. The full-length TCN grid must win.
    beats = np.arange(0.0, 1200.0, 0.5)
    monkeypatch.setattr(beat_models, "torch_available", lambda: True)
    monkeypatch.setattr(tcn, "detect_beats", lambda *a, **k: beats.copy())
    monkeypatch.setattr(refiner, "is_available", lambda: True)

    def _truncated_refine(*a, **k):
        return {
            "applied": True, "reason": "ok",
            "refined_beats": beats[beats < 900.0].tolist(),
            "refined_downbeats": beats[beats < 900.0][::4].tolist(),
            "covered_seconds": 903.0, "input_seconds": 1200.0,
        }

    monkeypatch.setattr(refiner, "refine", _truncated_refine)

    result = beat_models.detect_beat_grid("ignored.wav")
    assert result["backend"] == "tcn"
    assert result["refined"] is False
    assert np.isclose(result["beats"][-1], beats[-1])
    assert "kept full-length TCN grid" in result["notes"]


def test_detect_beat_grid_adopts_refiner_when_coverage_is_complete(monkeypatch):
    beats = np.arange(0.0, 200.0, 0.5)
    monkeypatch.setattr(beat_models, "torch_available", lambda: True)
    monkeypatch.setattr(tcn, "detect_beats", lambda *a, **k: beats.copy())
    monkeypatch.setattr(refiner, "is_available", lambda: True)
    monkeypatch.setattr(refiner, "refine", lambda *a, **k: {
        "applied": True, "reason": "ok",
        "refined_beats": (beats + 0.01).tolist(),
        "refined_downbeats": (beats + 0.01)[::4].tolist(),
        "covered_seconds": 200.0, "input_seconds": 200.0,
    })

    result = beat_models.detect_beat_grid("ignored.wav")
    assert result["backend"] == "tcn+refiner"
    assert result["refined"] is True
