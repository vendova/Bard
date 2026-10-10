"""Persistent render-job manager.

A render "job" records everything needed to resume an interrupted render after
an app restart (e.g. a Render.com container that exceeded its memory limit):

  input/render_jobs/<job_id>/
      job.json          metadata + settings + status + stage + seed
      job.log           appended log lines (live tail / replay)
      inputs/           copies of the uploaded audio + videos + zip
      segments/         one file per completed clip segment (resume store)
      beat_cache.pkl    pickled beat_info + selected_beats (stages 1-5)
      output.mp4        final video once the job is "done"

The manager is deliberately dependency-free (stdlib only) so it works in the
minimal portable environment.
"""

import hashlib
import json
import os
import pickle
import re
import shutil
import threading
import time
import uuid
from typing import Any, Dict, List, Tuple

from paths import get_state_dir

JOB_DIR_ROOT = os.path.join(get_state_dir(), "render_jobs")
_LOG_LOCK = threading.Lock()

# Statuses
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"


def _ensure_root() -> str:
    os.makedirs(JOB_DIR_ROOT, exist_ok=True)
    return JOB_DIR_ROOT


def _job_folder(job_id: str) -> str:
    return os.path.join(JOB_DIR_ROOT, job_id)


def _job_file(job_id: str) -> str:
    return os.path.join(_job_folder(job_id), "job.json")


def _log_file(job_id: str) -> str:
    return os.path.join(_job_folder(job_id), "job.log")


def _inputs_folder(job_id: str) -> str:
    return os.path.join(_job_folder(job_id), "inputs")


def _segments_folder(job_id: str) -> str:
    return os.path.join(_job_folder(job_id), "segments")


def _beat_cache_file(job_id: str) -> str:
    return os.path.join(_job_folder(job_id), "beat_cache.pkl")


def _output_file(job_id: str) -> str:
    return os.path.join(_job_folder(job_id), "output.mp4")


def _safe_copy(src: str, dest_dir: str) -> str:
    """Copy a file into dest_dir, returning the new path."""
    if not src or not os.path.isfile(src):
        return ""
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(src)
    # Avoid collisions / weird uploaded temp names
    if not name or name.startswith("gradio"):
        ext = os.path.splitext(src)[1] or ".bin"
        name = f"input_{int(time.time() * 1000) % 100000}{ext}"
    dest = os.path.join(dest_dir, name)
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy2(src, dest)
    return dest


def create_job(settings: Dict[str, Any]) -> str:
    """Create a new job and return its id."""
    job_id = uuid.uuid4().hex[:12]
    folder = _job_folder(job_id)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(_inputs_folder(job_id), exist_ok=True)
    os.makedirs(_segments_folder(job_id), exist_ok=True)
    record = {
        "job_id": job_id,
        "status": STATUS_RUNNING,
        "stage": "starting",
        "created": time.time(),
        "updated": time.time(),
        "seed": settings.get("seed"),
        "settings": settings,
        "inputs": settings.get("inputs", {}),
        "output_path": None,
    }
    _write_record(job_id, record)
    return job_id


def _write_record(job_id: str, record: Dict[str, Any]) -> None:
    record["updated"] = time.time()
    with open(_job_file(job_id), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)


def load_record(job_id: str) -> Dict[str, Any] | None:
    path = _job_file(job_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def update_job(job_id: str, **fields) -> None:
    record = load_record(job_id)
    if record is None:
        return
    record.update(fields)
    _write_record(job_id, record)


def append_log(job_id: str, line: str) -> None:
    if not job_id:
        return
    try:
        with _LOG_LOCK:
            with open(_log_file(job_id), "a", encoding="utf-8") as fh:
                fh.write(line.rstrip() + "\n")
    except Exception:
        pass


def read_log(job_id: str, max_lines: int = 200) -> List[str]:
    path = _log_file(job_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().splitlines()
        return lines[-max_lines:] if lines else []
    except Exception:
        return []


def persist_inputs(job_id: str, audio_file: str, video_files: List[str],
                   zip_file: str | None) -> Dict[str, Any]:
    """Copy uploaded files into the job folder so they survive restarts."""
    inputs_folder = _inputs_folder(job_id)
    audio_dest = _safe_copy(audio_file, inputs_folder) if audio_file else ""
    video_dests = [_safe_copy(v, inputs_folder) for v in (video_files or [])]
    video_dests = [v for v in video_dests if v]
    zip_dest = _safe_copy(zip_file, inputs_folder) if zip_file else ""
    return {
        "audio_file": audio_dest,
        "video_files": video_dests,
        "zip_file": zip_dest,
    }


def set_output(job_id: str, output_path: str) -> None:
    dest = _output_file(job_id)
    try:
        if output_path and os.path.isfile(output_path):
            if os.path.abspath(output_path) != os.path.abspath(dest):
                shutil.copy2(output_path, dest)
        update_job(job_id, status=STATUS_DONE, output_path=dest, stage="done")
    except Exception:
        update_job(job_id, status=STATUS_DONE, output_path=output_path, stage="done")


def get_output(job_id: str) -> str:
    record = load_record(job_id)
    if not record or record.get("status") != STATUS_DONE:
        return ""
    out = record.get("output_path") or ""
    if out and os.path.isfile(out):
        return out
    return ""


def mark_interrupted(job_id: str) -> None:
    if not job_id:
        return
    record = load_record(job_id)
    if record and record.get("status") == STATUS_RUNNING:
        update_job(job_id, status=STATUS_INTERRUPTED)


def mark_failed(job_id: str, message: str = "") -> None:
    if not job_id:
        return
    update_job(job_id, status=STATUS_FAILED, stage=f"failed: {message}"[:200])


# ── Beat-analysis cache ────────────────────────────────────────────────

_BEAT_CACHE_DIR = os.path.join(get_state_dir(), "beat_analysis_cache")


def _file_signature(path: str) -> str:
    """Stable signature of a file (name, size, mtime)."""
    try:
        st = os.stat(path)
        return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return f"{os.path.basename(path)}:missing"


def beat_cache_key(audio_file: str, video_files: List[str],
                   qwen_enabled: bool) -> str:
    sig = _file_signature(audio_file)
    for v in (video_files or []):
        sig += "|" + _file_signature(v)
    sig += f"|qwen={int(bool(qwen_enabled))}"
    return hashlib.sha1(sig.encode("utf-8", "ignore")).hexdigest()[:20]


def load_beat_cache(key: str) -> Tuple[Any, Any] | None:
    path = os.path.join(_BEAT_CACHE_DIR, f"{key}.pkl")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return data.get("selected_beats"), data.get("beat_info")
    except Exception:
        return None


def save_beat_cache(key: str, selected_beats: Any, beat_info: Any) -> None:
    os.makedirs(_BEAT_CACHE_DIR, exist_ok=True)
    path = os.path.join(_BEAT_CACHE_DIR, f"{key}.pkl")
    try:
        with open(path, "wb") as fh:
            pickle.dump({"selected_beats": selected_beats, "beat_info": beat_info}, fh)
    except Exception:
        pass


def save_job_beat_cache(job_id: str, selected_beats: Any, beat_info: Any) -> None:
    path = _beat_cache_file(job_id)
    try:
        with open(path, "wb") as fh:
            pickle.dump({"selected_beats": selected_beats, "beat_info": beat_info}, fh)
    except Exception:
        pass


def load_job_beat_cache(job_id: str) -> Tuple[Any, Any] | None:
    path = _beat_cache_file(job_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return data.get("selected_beats"), data.get("beat_info")
    except Exception:
        return None


# ── Segment store (per-clip resume) ────────────────────────────────────

def clip_signature(index: int, video_file: str, duration: float, fps: float,
                   target_size: Tuple[int, int] | None, use_nvenc: bool,
                   gpu_encoder: str, lossless: bool, seed: int,
                   effects_key: str = "") -> str:
    """Stable per-clip signature so a resumed render reuses finished segments."""
    parts = [
        f"i={index}",
        f"v={os.path.basename(video_file or '')}",
        f"d={float(duration):.4f}",
        f"fps={float(fps):.3f}",
        f"res={'x'.join(map(str, target_size)) if target_size else 'auto'}",
        f"nvenc={int(use_nvenc)}",
        f"enc={gpu_encoder}",
        f"lossless={int(lossless)}",
        f"seed={int(seed) if seed is not None else 0}",
        f"fx={effects_key}",
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8", "ignore")).hexdigest()[:16]


def get_segment(job_id: str, index: int, sig: str) -> str | None:
    """Return the path of a previously completed segment, or None."""
    path = os.path.join(_segments_folder(job_id), f"clip_{index:05d}_{sig}.mp4")
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    return None


def store_segment(job_id: str, index: int, sig: str, clip_path: str) -> str:
    """Persist a finished clip segment; return the stored path."""
    dest_dir = _segments_folder(job_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"clip_{index:05d}_{sig}.mp4")
    if clip_path and os.path.isfile(clip_path) and os.path.abspath(clip_path) != os.path.abspath(dest):
        try:
            shutil.copy2(clip_path, dest)
        except Exception:
            return clip_path
    return dest


def segment_counts(job_id: str, total: int) -> Tuple[int, int]:
    """Return (completed, total) segments for a job."""
    seg_dir = _segments_folder(job_id)
    if not os.path.isdir(seg_dir):
        return 0, total
    completed = len([f for f in os.listdir(seg_dir) if f.startswith("clip_") and f.endswith(".mp4")])
    return completed, total


def effects_config_key(effects_config) -> str:
    """Stable hash of an EffectsConfig so segment signatures reflect effect settings."""
    try:
        if effects_config is None:
            return "none"
        if hasattr(effects_config, "_asdict"):
            return hashlib.sha1(
                repr(sorted(effects_config._asdict().items())).encode("utf-8", "ignore")
            ).hexdigest()[:12]
        return hashlib.sha1(repr(effects_config).encode("utf-8", "ignore")).hexdigest()[:12]
    except Exception:
        return "unknown"


# ── Job discovery ──────────────────────────────────────────────────────

def latest_job() -> Dict[str, Any] | None:
    """Return the most recent job record (any status), or None."""
    _ensure_root()
    try:
        entries = os.listdir(JOB_DIR_ROOT)
    except OSError:
        return None
    records = []
    for entry in entries:
        if not re.fullmatch(r"[0-9a-f]{12}", entry):
            continue
        rec = load_record(entry)
        if rec:
            records.append(rec)
    if not records:
        return None
    records.sort(key=lambda r: r.get("updated") or r.get("created") or 0, reverse=True)
    return records[0]


def latest_resumable_job() -> Dict[str, Any] | None:
    """Return the most recent job that can be resumed (running/interrupted/failed)."""
    _ensure_root()
    try:
        entries = os.listdir(JOB_DIR_ROOT)
    except OSError:
        return None
    records = []
    for entry in entries:
        rec = load_record(entry)
        if rec and rec.get("status") in (STATUS_RUNNING, STATUS_INTERRUPTED, STATUS_FAILED):
            records.append(rec)
    if not records:
        return None
    records.sort(key=lambda r: r.get("updated") or r.get("created") or 0, reverse=True)
    return records[0]


def job_stage_label(job_id: str) -> str:
    """Human-friendly stage label for the resume UI."""
    rec = load_record(job_id)
    if not rec:
        return "unknown"
    status = rec.get("status", "unknown")
    stage = rec.get("stage", "")
    if status == STATUS_DONE:
        out = rec.get("output_path")
        if out and os.path.isfile(out):
            return f"✅ Completed — video ready ({os.path.getsize(out) / 1048576:.1f} MB)"
        return "✅ Completed"
    if status == STATUS_RUNNING:
        return f"⏳ In progress — {stage}" if stage else "⏳ In progress"
    if status == STATUS_INTERRUPTED:
        return f"⚠️ Interrupted (likely restart) — {stage}" if stage else "⚠️ Interrupted"
    if status == STATUS_FAILED:
        return f"❌ Failed — {stage}" if stage else "❌ Failed"
    return stage or status


def trim_old_jobs(keep: int = 8) -> None:
    """Keep only the most recent `keep` jobs to avoid unbounded disk growth."""
    _ensure_root()
    try:
        entries = os.listdir(JOB_DIR_ROOT)
    except OSError:
        return
    records = []
    for entry in entries:
        if not re.fullmatch(r"[0-9a-f]{12}", entry):
            continue
        path = _job_file(entry)
        if os.path.isfile(path):
            try:
                records.append((os.path.getmtime(path), entry))
            except OSError:
                pass
    if len(records) <= keep:
        return
    records.sort(key=lambda x: x[0], reverse=True)
    for _, job_id in records[keep:]:
        shutil.rmtree(_job_folder(job_id), ignore_errors=True)
