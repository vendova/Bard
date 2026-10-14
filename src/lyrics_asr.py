"""
Optional speech-to-text for lyric transcription.

Transcribes a music track into time-stamped ``{start, end, text}`` segments
using whichever ASR backend is installed, in order of preference:

1. ``faster-whisper`` (CTranslate2 — CPU-friendly, no PyTorch)
2. ``openai-whisper``
3. ``vosk`` (needs a model directory via ``BEATSYNC_VOSK_MODEL``)

Nothing here is bundled and no model is downloaded automatically. When no
backend is available the callers get a clear message and the user can still
paste lyrics by hand. Install one with, e.g.::

    pip install -r requirements-lyrics.txt   # faster-whisper

Model size is chosen with ``BEATSYNC_WHISPER_MODEL`` (default ``large-v3`` —
the highest-accuracy multilingual model, best for Indian languages). Smaller
models (``tiny``/``base``/``small``/``medium``) trade accuracy for speed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

_BACKEND_ORDER = ("faster_whisper", "whisper", "vosk")

# Whisper model cache so repeated transcribes do not reload weights.
_MODEL_CACHE: Dict[str, object] = {}


def available_backend() -> Optional[str]:
    """Return the name of the first installed ASR backend, or ``None``."""
    for module in _BACKEND_ORDER:
        if importlib.util.find_spec(module) is not None:
            return module
    return None


def is_available() -> bool:
    return available_backend() is not None


def _model_size() -> str:
    return os.environ.get("BEATSYNC_WHISPER_MODEL", "large-v3").strip() or "large-v3"


def _compute_type() -> str:
    # int8 is the CPU-friendly default and keeps good accuracy; override with
    # BEATSYNC_WHISPER_COMPUTE=float32 for maximum fidelity (much slower).
    return os.environ.get("BEATSYNC_WHISPER_COMPUTE", "int8").strip() or "int8"


def transcribe(audio_file: str, language: Optional[str] = None,
               model_size: Optional[str] = None) -> List[Dict[str, object]]:
    """Transcribe ``audio_file`` into ``{start, end, text}`` segments.

    ``language`` is an ISO code (e.g. ``"hi"``) or ``None`` to auto-detect.
    Raises ``RuntimeError`` when no backend is installed.
    """
    backend = available_backend()
    if backend is None:
        raise RuntimeError(
            "No speech-to-text backend installed. Install faster-whisper "
            "(pip install -r requirements-lyrics.txt) or paste lyrics manually."
        )
    size = model_size or _model_size()
    if backend == "faster_whisper":
        return _transcribe_faster_whisper(audio_file, language, size)
    if backend == "whisper":
        return _transcribe_whisper(audio_file, language, size)
    return _transcribe_vosk(audio_file, language)


def _decode_pcm(audio_file: str) -> "object":
    """Decode any media to 16 kHz mono float32 via FFmpeg.

    faster-whisper's built-in loader uses PyAV, whose API drifts between
    releases; decoding with FFmpeg directly keeps us independent of that and
    matches the FFmpeg binary the rest of the app already resolves.
    """
    import numpy as np

    from media_paths import resolve_ffmpeg  # type: ignore

    ffmpeg, _ = resolve_ffmpeg()  # type: ignore
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", audio_file,
        "-f", "f32le", "-ac", "1", "-ar", "16000", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Audio decode failed: " + (proc.stderr.decode("utf-8", "ignore")[-300:])
        )
    return np.frombuffer(proc.stdout, dtype="<f4").astype("float32")


def _transcribe_faster_whisper(audio_file: str, language: Optional[str],
                               model_size: str) -> List[Dict[str, object]]:
    from faster_whisper import WhisperModel

    key = f"{model_size}:{_compute_type()}"
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type=_compute_type())
        _MODEL_CACHE[key] = model
    audio = _decode_pcm(audio_file)
    segments, _info = model.transcribe(
        audio, language=language or None, vad_filter=True,
        condition_on_previous_text=False, beam_size=5,
    )
    return [
        {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
        for s in segments
        if (s.text or "").strip()
    ]


def _transcribe_whisper(audio_file: str, language: Optional[str],
                        model_size: str) -> List[Dict[str, object]]:
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file, language=language or None)
    return [
        {"start": float(seg["start"]), "end": float(seg["end"]),
         "text": (seg.get("text") or "").strip()}
        for seg in result.get("segments", [])
        if (seg.get("text") or "").strip()
    ]


def _transcribe_vosk(audio_file: str, language: Optional[str]) -> List[Dict[str, object]]:
    import json
    import wave

    from vosk import KaldiRecognizer, Model

    model_dir = os.environ.get("BEATSYNC_VOSK_MODEL", "").strip()
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError(
            "Vosk is installed but no model is set. Point BEATSYNC_VOSK_MODEL "
            "at an extracted Vosk model directory."
        )

    # Vosk wants 16 kHz mono PCM WAV.
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", audio_file,
             "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav],
            check=True,
        )
        model = Model(model_dir)
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        segments: List[Dict[str, object]] = []
        with wave.open(tmp_wav, "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = (res.get("text") or "").strip()
                    words = res.get("result") or []
                    if text and words:
                        segments.append({
                            "start": float(words[0]["start"]),
                            "end": float(words[-1]["end"]),
                            "text": text,
                        })
        final = json.loads(rec.FinalResult())
        text = (final.get("text") or "").strip()
        words = final.get("result") or []
        if text and words:
            segments.append({
                "start": float(words[0]["start"]),
                "end": float(words[-1]["end"]),
                "text": text,
            })
        return segments
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
