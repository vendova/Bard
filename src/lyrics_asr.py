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

# Selectable Whisper models, smallest/fastest first. large-v3 is the most
# accurate (best for Indic languages); turbo is a good speed/accuracy balance.
# HuggingFace repo ids are also allowed: a fine-tune published in the
# Transformers (PyTorch) format is converted to CTranslate2 on first use and
# cached, so it runs through the same fast CPU path as the stock sizes.
MODEL_CHOICES = [
    ("tiny — fastest, lowest accuracy", "tiny"),
    ("base — fast, basic accuracy", "base"),
    ("small — balanced", "small"),
    ("medium — high accuracy", "medium"),
    ("large-v3 — highest accuracy (recommended)", "large-v3"),
    ("large-v3-turbo — high accuracy, faster", "large-v3-turbo"),
    ("Telugu — vasista22/whisper-telugu-large-v2 (fine-tuned)", "vasista22/whisper-telugu-large-v2"),
    ("Telugu — lamy/Telugu-transcription (fine-tuned)", "lamyer/Telugu-transcription"),
]
DEFAULT_MODEL = "large-v3"

# Cache dir for CTranslate2 conversions of HuggingFace Transformers models.
_CONVERTED_DIR = os.environ.get(
    "BEATSYNC_ASR_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "beatsync_asr")
)


def available_backend() -> Optional[str]:
    """Return the name of the first installed ASR backend, or ``None``."""
    for module in _BACKEND_ORDER:
        if importlib.util.find_spec(module) is not None:
            return module
    return None


def is_available() -> bool:
    return available_backend() is not None


def _model_size() -> str:
    return os.environ.get("BEATSYNC_WHISPER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _compute_type() -> str:
    # int8 is the CPU-friendly default and keeps good accuracy; override with
    # BEATSYNC_WHISPER_COMPUTE=float32 for maximum fidelity (much slower).
    return os.environ.get("BEATSYNC_WHISPER_COMPUTE", "int8").strip() or "int8"


def transcribe(audio_file: str, language: Optional[str] = None,
               model_size: Optional[str] = None, progress=None) -> List[Dict[str, object]]:
    """Transcribe ``audio_file`` into ``{start, end, text}`` segments.

    ``language`` is an ISO code (e.g. ``"hi"``) or ``None`` to auto-detect.
    ``progress`` is an optional ``callable(str)`` that receives live status
    lines (model download/load, decode, per-segment progress).
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
        return _transcribe_faster_whisper(audio_file, language, size, progress)
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


def _looks_like_hf_repo(model: str) -> bool:
    """A HuggingFace repo id like ``owner/name`` (not a local path)."""
    if not model or "/" not in model:
        return False
    if os.path.isabs(model) or os.path.exists(model):
        return False
    owner, _, name = model.partition("/")
    return bool(owner and name) and not name.startswith((".", "/"))


def _hf_model_language(repo: str) -> Optional[str]:
    """Best-effort language code for known fine-tuned repos."""
    low = repo.lower()
    if "telugu" in low or "te" == low.split("/")[-1][:2]:
        return "te"
    return None


def _resolve_faster_whisper_model(model_size: str, progress=None) -> str:
    """Return a path/name faster-whisper can load, converting HF repos if needed.

    Stock sizes ("large-v3", ...) are returned unchanged. A HuggingFace repo id
    in the Transformers (PyTorch) format is converted to CTranslate2 once and
    the result is cached under ``_CONVERTED_DIR``.
    """
    if not _looks_like_hf_repo(model_size):
        return model_size

    safe = model_size.replace("/", "__")
    dest = os.path.join(_CONVERTED_DIR, safe)
    if os.path.isdir(dest) and os.path.exists(os.path.join(dest, "model.bin")):
        return dest

    def log(msg: str) -> None:
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass

    log(f"Converting '{model_size}' to CTranslate2 (one-time; needs torch+transformers)…")
    import ctranslate2

    os.makedirs(_CONVERTED_DIR, exist_ok=True)
    # float16 + low-memory loading keeps peak RAM bounded for large checkpoints
    # (a full large-v2 conversion can otherwise exhaust a small CPU box).
    converter = ctranslate2.converters.TransformersConverter(
        model_size, load_as_float16=True, low_cpu_mem_usage=True
    )
    converter.convert(dest, quantization="int8", force=False)
    log(f"Conversion complete: {dest}")
    return dest


def _transcribe_faster_whisper(audio_file: str, language: Optional[str],
                               model_size: str, progress=None) -> List[Dict[str, object]]:
    from faster_whisper import WhisperModel

    def log(msg: str) -> None:
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass

    resolved = _resolve_faster_whisper_model(model_size, progress)
    if language is None:
        language = _hf_model_language(model_size)

    key = f"{resolved}:{_compute_type()}"
    model = _MODEL_CACHE.get(key)
    if model is None:
        log(f"Loading Whisper model '{model_size}' (downloads once, then cached)…")
        model = WhisperModel(resolved, device="cpu", compute_type=_compute_type())
        _MODEL_CACHE[key] = model
    else:
        log(f"Using cached Whisper model '{model_size}'")

    log("Decoding audio to 16 kHz mono…")
    audio = _decode_pcm(audio_file)
    log(f"Decoded {len(audio) / 16000:.1f}s of audio; transcribing…")

    segments, info = model.transcribe(
        audio, language=language or None, vad_filter=True,
        condition_on_previous_text=False, beam_size=5,
    )
    detected = getattr(info, "language", None)
    if detected:
        log(f"Detected speech language: {detected}")
    out: List[Dict[str, object]] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        out.append({"start": float(s.start), "end": float(s.end), "text": text})
        log(f"  [{s.start:6.2f}s] {text[:70]}")
    log(f"Transcribed {len(out)} line(s)")
    return out


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
