import os
import sys
import contextlib
import asyncio

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def _install_windows_asyncio_connection_reset_filter() -> None:
    """Hide benign Windows asyncio pipe resets after browser/subprocess shutdown.

    On Windows, asyncio's Proactor transport can log a scary traceback when a
    local socket or subprocess pipe is closed by the other side after the real
    work is already complete. The app result is not affected, so suppress only
    that exact WinError 10054 callback and let all other async errors through.
    """
    if os.name != "nt" or getattr(asyncio, "_beatsync_win10054_filter", False):
        return
    asyncio._beatsync_win10054_filter = True

    def is_benign_reset(exc: BaseException | None) -> bool:
        if not isinstance(exc, ConnectionResetError):
            return False
        winerror = getattr(exc, "winerror", None)
        errno_value = getattr(exc, "errno", None)
        return winerror == 10054 or errno_value == 10054 or "WinError 10054" in str(exc)

    # Directly patch the noisy Proactor pipe cleanup callback when available.
    try:
        from asyncio import proactor_events

        transport_cls = getattr(proactor_events, "_ProactorBasePipeTransport", None)
        original_call_lost = getattr(transport_cls, "_call_connection_lost", None)
        if transport_cls is not None and original_call_lost is not None:
            def quiet_call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
                try:
                    return original_call_lost(self, exc)
                except ConnectionResetError as reset_exc:
                    if is_benign_reset(reset_exc):
                        return None
                    raise

            transport_cls._call_connection_lost = quiet_call_connection_lost
    except Exception:
        pass

    # Fallback for the same exception if it still reaches the loop logger.
    original_exception_handler = asyncio.BaseEventLoop.call_exception_handler

    def quiet_exception_handler(self, context):  # type: ignore[no-untyped-def]
        exc = context.get("exception") if isinstance(context, dict) else None
        handle = str(context.get("handle", "")) if isinstance(context, dict) else ""
        if is_benign_reset(exc) and "_ProactorBasePipeTransport._call_connection_lost" in handle:
            return None
        return original_exception_handler(self, context)

    asyncio.BaseEventLoop.call_exception_handler = quiet_exception_handler


_install_windows_asyncio_connection_reset_filter()

from logger import (
    setup_environment,
    USING_PORTABLE_PYTHON, USING_PORTABLE_CUDA, USING_CUPY_CTK, FFMPEG_FOUND
)

# Initialize environment
setup_environment()
# NOW import other modules (after CUDA environment is set)
import gradio as gr
import tempfile
import shutil
import datetime
import multiprocessing
import queue
import re
import subprocess
import threading
import time
import socket
from typing import Callable, Iterator, TypeAlias, Tuple, Dict, List

# Import FFmpeg processing module
from ffmpeg_processing import get_video_fps, FFMPEG_PATH, compress_video_to_target_size, optimize_for_web

# Shared runtime settings
from gpu_cpu_utils import (
    CPU_COUNT,
    MAX_THREADS,
    PARALLEL_WORKERS,
    GPU_INFO,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    set_gpu_mode,
)
from paths import (
    GRADIO_TEMP_DIR,
    get_input_dir,
    get_audio_input_dir,
    get_video_input_dir,
    get_output_dir,
)

gpu_data = GPU_INFO
gpu_info = f"{gpu_data['name']} ({gpu_data['cuda_version']})" if gpu_data['available'] else "CPU Mode"

from video_processor import create_music_video
from video_effects import EffectsConfig, parse_effects_from_ui, TRANSITION_TYPES

from font_presets import get_combo_names as _get_font_combo_names
from transition_presets import get_preset_names as _get_transition_preset_names

# Build UI dropdown choices with a "Random" option at top
_FONT_COMBO_CHOICES = [("🎲 Random (new combo each render)", "")] + [(n, n) for n in _get_font_combo_names()]
_TRANSITION_PRESET_CHOICES = [("🎲 Random (new preset each render)", "")] + [(n, n) for n in _get_transition_preset_names()]

from auto_mode import analyze_beats_auto

# Import UI content
from ui_content import *

# ── Qwen backend availability probe ────────────────────────────────────
# Used to smartly enable/disable the semantic-detection toggle in the UI.
try:
    from video_analysis import _qwen_backend_available, DEFAULT_QWEN_MODEL_DIR
    QWEN_BACKEND_AVAILABLE = bool(_qwen_backend_available(DEFAULT_QWEN_MODEL_DIR))
except Exception:
    QWEN_BACKEND_AVAILABLE = False
if os.environ.get("BEATSYNC_DISABLE_QWEN", "0") == "1":
    QWEN_BACKEND_AVAILABLE = False

# Persistent render-job manager (resume after restart / memory-limit crash).
try:
    import render_jobs as _jobs
except Exception:
    _jobs = None

# Optional hard cap on parallel clip workers — lowers memory on hosts like
# Render.com that enforce a RAM ceiling. Set BEATSYNC_MAX_PARALLEL_WORKERS.
try:
    _env_workers = os.environ.get("BEATSYNC_MAX_PARALLEL_WORKERS", "").strip()
    if _env_workers:
        _capped = max(1, int(_env_workers))
        if _capped < PARALLEL_WORKERS:
            PARALLEL_WORKERS = _capped
            print(f"⚠️ BEATSYNC_MAX_PARALLEL_WORKERS={_capped} → parallel workers capped to {_capped}")
except Exception:
    pass

# Set environment variable for Gradio
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR

VideoFilesInput : TypeAlias = List[str]
StatusResult : TypeAlias = Tuple[str, str, Dict, object]

STATUS_BOX_CSS = """
#status-output-box {
    min-height: 420px !important;
}

#status-output-box textarea {
    height: 400px !important;
    min-height: 400px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    resize: none !important;
    font-family: 'SFMono-Regular', 'Cascadia Code', 'Consolas', 'Liberation Mono', monospace !important;
    font-size: 12.5px !important;
    line-height: 1.55 !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
}
"""


# Auto-click the download button once a render finishes so the MP4 begins
# downloading immediately without the user having to press anything. Retries
# briefly because Gradio applies the visibility/value update asynchronously.
_AUTO_DOWNLOAD_JS = """
() => {
  const tryClick = (n) => {
    const el = document.getElementById('download-video-btn');
    if (!el) { if (n < 25) setTimeout(() => tryClick(n + 1), 200); return; }
    const btn = el.querySelector('button') || el;
    if (btn && !btn.disabled && btn.offsetParent !== null) { btn.click(); return; }
    if (n < 25) setTimeout(() => tryClick(n + 1), 200);
  };
  tryClick(0);
}
"""


# Persist lightweight settings to localStorage so an accidental refresh does
# not lose the user's configuration. Gradio resets component values on reload,
# so we snapshot labelled inputs and restore them once the app is ready.
_SETTINGS_PERSISTENCE_JS = """
() => {
  const KEY = 'beatsync_settings_v1';
  const RESTORE_KEY = 'beatsync_pending_restore';
  const pick = () => {
    const out = {};
    document.querySelectorAll('fieldset, .gradio-container label').forEach((fs) => {
      const labelEl = fs.querySelector('span[data-testid="block-info"], .label-wrap span, label > span');
      const input = fs.querySelector('input, textarea, select');
      if (!input || !labelEl) return;
      const name = (labelEl.textContent || '').trim();
      if (!name) return;
      try {
        if (input.type === 'checkbox') out[name] = { v: input.checked, t: 'c' };
        else if (input.type === 'radio') { if (input.checked) out[name] = { v: input.value, t: 'r' }; }
        else out[name] = { v: input.value, t: input.tagName.toLowerCase() };
      } catch (e) {}
    });
    return out;
  };
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(pick())); } catch (e) {} };
  const apply = (data) => {
    document.querySelectorAll('fieldset, .gradio-container label').forEach((fs) => {
      const labelEl = fs.querySelector('span[data-testid="block-info"], .label-wrap span, label > span');
      const input = fs.querySelector('input, textarea, select');
      if (!input || !labelEl) return;
      const name = (labelEl.textContent || '').trim();
      const entry = data[name];
      if (!entry) return;
      try {
        if (entry.t === 'c') { if (input.checked !== entry.v) input.click(); }
        else if (entry.t === 'r') { if (input.value === entry.v && !input.checked) input.click(); }
        else {
          const proto = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
          const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
          setter.call(input, entry.v);
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
      } catch (e) {}
    });
  };
  // Restore once per page load, after Gradio finishes the first render.
  if (!window.__beatsyncRestoreDone) {
    window.__beatsyncRestoreDone = true;
    const raw = localStorage.getItem(KEY);
    if (raw) {
      let tries = 0;
      const attempt = () => {
        try { apply(JSON.parse(raw)); } catch (e) {}
        if (++tries < 8) setTimeout(attempt, 400);
      };
      setTimeout(attempt, 600);
    }
  }
  if (!window.__beatsyncSaveHooked) {
    window.__beatsyncSaveHooked = true;
    document.addEventListener('change', save, true);
    document.addEventListener('input', (e) => { if (e.target && e.target.tagName !== 'TEXTAREA') save(); }, true);
    window.addEventListener('beforeunload', save);
  }
}
"""



def _stage_status(stage_number: int) -> str:
    return f"Stage {stage_number} is processing. Please wait."


# ── Resume-after-restart helpers ──────────────────────────────────────

def _resume_status_markdown() -> str:
    """Describe the most recent resumable job for the Resume panel."""
    if _jobs is None:
        return "_Job persistence unavailable._"
    rec = _jobs.latest_resumable_job()
    if not rec:
        return "_No interrupted render to resume yet._"
    job_id = rec.get("job_id", "")
    label = _jobs.job_stage_label(job_id)
    created = rec.get("created")
    when = ""
    if created:
        when = datetime.datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
    return f"**Job `{job_id}`** — {label}\n\n_Started {when}_"


def _load_resume_panel():
    """Backend load: reveal the Resume panel when an unfinished job exists."""
    if _jobs is None:
        return gr.update(visible=False), _resume_status_markdown()
    rec = _jobs.latest_resumable_job()
    return gr.update(visible=bool(rec)), _resume_status_markdown()


def _resume_last_job():
    """Click handler: reconnect to the most recent interrupted job.

    Yields the same StatusResult shape as process_video so the existing
    outputs (video, status box, session, download button) are reused.
    """
    if _jobs is None:
        yield None, "❌ Job persistence unavailable.", {}, gr.update(visible=False)
        return
    rec = _jobs.latest_resumable_job()
    if not rec:
        yield None, "ℹ️ Nothing to resume — no interrupted render found.", {}, gr.update(visible=False)
        return
    job_id = rec["job_id"]
    # If the job actually finished before the restart, just hand back the video.
    done = _jobs.get_output(job_id)
    if done:
        yield done, f"✅ Job {job_id} already completed — video ready.", {}, gr.update(visible=True, value=done)
        return
    rec = _jobs.load_record(job_id) or {}
    inp = rec.get("inputs") or {}
    yield from process_video(
        audio_file=inp.get("audio_file", ""),
        video_files=inp.get("video_files", []),
        output_filename=(rec.get("settings") or {}).get("output_filename", "music_video.mp4"),
        processing_mode=(rec.get("settings") or {}).get("processing_mode", "cpu"),
        custom_fps=None,
        session_state={"_resume_seed": rec.get("seed")},
        zip_file=inp.get("zip_file") or None,
        smart_sync=bool((rec.get("settings") or {}).get("smart_sync", False)),
        qwen_semantics=bool((rec.get("settings") or {}).get("qwen_semantics", True)),
        resume_job_id=job_id,
    )



class LiveLogCapture:
    """Tee stdout to a real sink (server log) while streaming complete
    lines to a callback so they appear live in the Gradio status box."""

    def __init__(self, sink, on_line):
        self.sink = sink
        self.on_line = on_line
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self.sink is not None:
            try:
                self.sink.write(text)
            except Exception:
                pass
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                try:
                    self.on_line(line)
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        if self.sink is not None:
            try:
                self.sink.flush()
            except Exception:
                pass


class StageConsoleLogger:
    """Small CMD logger: stage start, up to 5 useful lines, stage end."""

    def __init__(self, stream, max_lines_per_stage: int = 5):
        self.stream = stream
        self.max_lines_per_stage = max(1, int(max_lines_per_stage))
        self.stage_number: int | None = None
        self.stage_started = 0.0
        self.stage_line_count = 0
        self.total_started = time.perf_counter()

    def start_stage(self, stage_number: int) -> None:
        if self.stage_number == stage_number:
            return
        self.end_stage()
        self.stage_number = stage_number
        self.stage_started = time.perf_counter()
        self.stage_line_count = 0
        self._write(f"Stage {stage_number} processing started:\n")

    def stage_line(self, stage_number: int, message: str) -> None:
        if self.stage_number != stage_number:
            self.start_stage(stage_number)
        self.line(message)

    def line(self, message: str) -> None:
        if self.stage_number is None:
            return
        if self.stage_line_count >= self.max_lines_per_stage:
            return
        message = self._clean(message)
        if message:
            self._write(f"  {message}\n")
            self.stage_line_count += 1

    def end_stage(self) -> None:
        if self.stage_number is None:
            return
        elapsed = int(round(time.perf_counter() - self.stage_started))
        self._write(f"Stage {self.stage_number} ended in {elapsed} seconds.\n\n")
        self.stage_number = None
        self.stage_started = 0.0
        self.stage_line_count = 0

    def finish(self) -> None:
        self.end_stage()
        elapsed = int(round(time.perf_counter() - self.total_started))
        self._write(f"Total time processing: {elapsed} seconds\n")

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clean(self, text: str) -> str:
        text = str(text).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", text).strip()


def _fmt_stage_seconds(seconds: float | int | None) -> str:
    try:
        return f"{float(seconds):.1f}s"
    except Exception:
        return "0.0s"


def _short_model_name(model_id: str | None) -> str:
    if not model_id:
        return ""
    return os.path.basename(str(model_id).rstrip("/\\")) or str(model_id)


def _stage5_summary(console_logger: StageConsoleLogger | None, video_analysis: Dict | None) -> None:
    if console_logger is None or not isinstance(video_analysis, dict):
        return

    source_count = int(video_analysis.get("source_count") or len(video_analysis.get("videos") or []))
    worker_count = int(video_analysis.get("worker_count") or 1)
    cache_hits = int(video_analysis.get("cache_hits") or 0)
    ai_enabled = bool(video_analysis.get("ai_enabled"))
    qwen_total = int(video_analysis.get("qwen_frame_count") or 0)
    qwen_tags = int(video_analysis.get("qwen_tag_count") or 0)
    model_id = _short_model_name(video_analysis.get("qwen_model_id"))
    batch_size = int(video_analysis.get("qwen_concurrency") or 0)

    console_logger.line(f"Source videos: {source_count}, visual workers: {worker_count}")
    if ai_enabled:
        qwen_bits = ["Qwen: enabled"]
        if model_id:
            qwen_bits.append(f"model {model_id}")
        if batch_size:
            qwen_bits.append(f"batch {batch_size}")
        console_logger.line(", ".join(qwen_bits))
        if qwen_total:
            inference_seconds = float(video_analysis.get("qwen_inference_seconds") or video_analysis.get("qwen_seconds") or 0.0)
            qwen_rate = (qwen_total / inference_seconds) if inference_seconds > 0 else 0.0
            peak_vram = float(video_analysis.get("qwen_peak_vram_gb") or 0.0)
            perf_bits = []
            if batch_size:
                perf_bits.append(f"batch {batch_size}")
            if peak_vram > 0:
                perf_bits.append(f"~{peak_vram:.2f} GB VRAM")
            if qwen_rate > 0:
                perf_bits.append(f"{qwen_rate:.2f} candidates/s")
            if perf_bits:
                console_logger.line(f"Qwen performance: {', '.join(perf_bits)}")
            console_logger.line(
                f"Qwen tags: {qwen_tags}/{qwen_total} in {_fmt_stage_seconds(video_analysis.get('qwen_seconds'))}"
            )
    else:
        console_logger.line("Qwen: disabled")

    summary = video_analysis.get("summary")
    if summary:
        console_logger.line(f"Visual library: {summary}")
    console_logger.line(
        f"Analysis time: {_fmt_stage_seconds(video_analysis.get('analysis_seconds'))}, cache {cache_hits}/{source_count}"
    )


def _stage6_summary(console_logger: StageConsoleLogger | None, beat_info: Dict | None) -> None:
    if console_logger is None or not isinstance(beat_info, dict):
        return

    render_info = beat_info.get("render_info") or {}
    if not render_info:
        return

    cuts = int(render_info.get("render_cuts") or 0)
    frames = int(render_info.get("timeline_frames") or 0)
    fps = render_info.get("output_fps")
    if cuts or frames:
        fps_text = f" @ {float(fps):.1f} FPS" if fps is not None else ""
        console_logger.line(f"Render timeline: {cuts} cuts, {frames} frames{fps_text}")

    clip_workers = render_info.get("clip_workers")
    requested_workers = render_info.get("requested_workers")
    worker_text = ""
    if clip_workers:
        worker_text = f", workers {clip_workers}"
        if requested_workers and requested_workers != clip_workers:
            worker_text += f"/{requested_workers}"
    encoder = render_info.get("encoder")
    if encoder or worker_text:
        console_logger.line(f"Encoder: {encoder or 'unknown'}{worker_text}")

    if render_info.get("audio_duration") is not None:
        console_logger.line(f"Audio duration: {float(render_info['audio_duration']):.2f} seconds")

    plan_summary = render_info.get("plan_summary") or {}
    if plan_summary:
        console_logger.line(
            "Planner: "
            f"{int(plan_summary.get('clip_count') or 0)} clips, "
            f"{int(plan_summary.get('source_count') or 0)} sources, "
            f"AI moments {int(plan_summary.get('ai_tagged') or 0)}"
        )

    final_bits = []
    if render_info.get("target_resolution"):
        final_bits.append(f"resolution {render_info['target_resolution']}")
    if render_info.get("final_assembly_seconds") is not None:
        final_bits.append(f"assembly {_fmt_stage_seconds(render_info['final_assembly_seconds'])}")
    if final_bits:
        console_logger.line("Final: " + ", ".join(final_bits))

def find_launch_port(default_port: int = 7860, search_limit: int = 20) -> int:
    """Prefer the default Gradio port, then step forward if it is busy."""
    # Render and most PaaS providers set PORT; respect it above all else.
    for env_var in ("GRADIO_SERVER_PORT", "PORT"):
        env_port = os.environ.get(env_var)
        if env_port:
            try:
                return int(env_port)
            except ValueError:
                print(f"⚠️ Invalid {env_var}={env_port!r}; using auto port search.")

    for port in range(default_port, default_port + search_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                if port != default_port:
                    print(f"⚠️ Port {default_port} is busy. Starting BeatSync on port {port}.")
                return port

    raise OSError(f"Cannot find empty port in range: {default_port}-{default_port + search_limit - 1}")


def _as_existing_source_path(file_path: str | None) -> str | None:
    """Use the selected source file directly instead of copying it locally."""
    if not file_path:
        return None
    try:
        path = os.path.abspath(os.fspath(file_path))
    except TypeError:
        return None
    return path if os.path.isfile(path) else None


def _as_existing_source_paths(file_paths: VideoFilesInput) -> list[str]:
    if not file_paths:
        return []
    return [path for path in (_as_existing_source_path(p) for p in file_paths) if path]


_ZIP_VIDEO_EXTENSIONS = ('.mp4', '.mkv')


def _is_zip_path_traversal(name: str) -> bool:
    """True if a zip entry name escapes its extraction root (zip-slip)."""
    norm = os.path.normpath(name)
    if os.path.isabs(norm) or norm.startswith('..'):
        return True
    parts = norm.split(os.sep)
    return any(part == '..' for part in parts)


def _extract_zip_clips(zip_path: str, dest_dir: str) -> list[str]:
    """Extract video clips from a zip archive into dest_dir.

    Returns a sorted list of extracted clip paths. Only files matching the
    supported video extensions are extracted; nested directory structures are
    flattened into dest_dir. Zip-slip entries are rejected.
    """
    import zipfile

    if not zip_path or not os.path.isfile(zip_path):
        return []

    os.makedirs(dest_dir, exist_ok=True)
    extracted: list[str] = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            members = [
                m for m in archive.infolist()
                if not m.is_dir() and os.path.splitext(m.filename)[1].lower() in _ZIP_VIDEO_EXTENSIONS
            ]
            for member in members:
                if _is_zip_path_traversal(member.filename):
                    continue
                # Flatten: keep only the basename to avoid collisions / deep trees.
                base = os.path.basename(member.filename) or 'clip'
                out_path = os.path.join(dest_dir, base)
                # Avoid overwriting when multiple clips share a basename.
                if os.path.exists(out_path):
                    stem, ext = os.path.splitext(base)
                    counter = 1
                    while os.path.exists(out_path):
                        out_path = os.path.join(dest_dir, f"{stem}_{counter}{ext}")
                        counter += 1
                with archive.open(member, 'r') as src, open(out_path, 'wb') as dst:
                    dst.write(src.read())
                extracted.append(out_path)
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"   ⚠️  Could not extract zip {zip_path}: {exc}")
        return []

    extracted.sort(key=lambda p: os.path.basename(p).lower())
    return extracted


def _collect_video_clips(video_files: VideoFilesInput, zip_file: str | None,
                         session_dir: str, console_logger: StageConsoleLogger | None) -> list[str]:
    """Combine individually uploaded clips with clips extracted from a zip."""
    clips = _as_existing_source_paths(video_files)
    if zip_file:
        zip_dest = os.path.join(session_dir, 'zip_clips')
        extracted = _extract_zip_clips(zip_file, zip_dest)
        if extracted and console_logger:
            console_logger.line(f"🗜️  Extracted {len(extracted)} clips from zip")
        elif console_logger:
            console_logger.line("⚠️  No video clips (.mp4/.mkv) found inside the zip")
        clips.extend(extracted)
    return clips


def _process_video_impl(audio_file: str, video_files: VideoFilesInput,
                       output_filename: str, processing_mode: str,
                       custom_fps: float, session_state: dict,
                       effects_config: EffectsConfig = None,
                       smart_sync: bool = False,
                       lyrics_text: str = "",
                       font_combo_name: str = "",
                       transition_preset_name: str = "",
                       progress_callback: Callable[[str], None] | None = None,
                       console_logger: StageConsoleLogger | None = None,
                       regenerate: bool = False,
                       zip_file: str | None = None,
                       qwen_semantics: bool = True,
                       job_id: str | None = None) -> StatusResult:
    total_started = time.perf_counter()
    try:
        parallel_workers = PARALLEL_WORKERS

        # Initialize session state if needed
        if 'original_audio_path' not in session_state:
            session_state['original_audio_path'] = None
            session_state['original_video_paths'] = []
        if 'session_dir' not in session_state or not os.path.isdir(session_state['session_dir']):
            session_state['session_dir'] = tempfile.mkdtemp(prefix='beatsync_', dir=GRADIO_TEMP_DIR)
        session_dir = session_state['session_dir']

        # Handle audio by referencing the selected file path directly.
        if audio_file:
            if audio_file != session_state.get('original_audio_path'):
                local_audio_path = _as_existing_source_path(audio_file)
                if local_audio_path:
                    session_state['local_audio_path'] = local_audio_path
                    session_state['original_audio_path'] = audio_file
                else:
                    return None, '❌ Error: Could not access audio file', session_state
            else:
                local_audio_path = session_state.get('local_audio_path')
        else:
            return None, '❌ Error: No audio file selected', session_state

        # Handle videos: combine individually uploaded clips with any clips
        # extracted from an optional zip archive.
        sources_key = (tuple(video_files or ()), zip_file)
        if sources_key != session_state.get('original_video_paths'):
            local_video_paths = _collect_video_clips(
                video_files, zip_file, session_dir, console_logger
            )
            if local_video_paths:
                session_state['local_video_paths'] = local_video_paths
                session_state['original_video_paths'] = sources_key
            else:
                return None, '❌ Error: No video clips available. Upload clips or a zip of clips.', session_state
        else:
            local_video_paths = session_state.get('local_video_paths')

        # Verify files exist
        if not local_audio_path or not os.path.exists(local_audio_path):
             return None, f"❌ Error: Audio file is missing or inaccessible.", session_state
        if not local_video_paths or not all(p and os.path.exists(p) for p in local_video_paths):
             return None, f"❌ Error: Video files are missing or inaccessible.", session_state
        
        # Set GPU mode
        use_gpu = GPU_AVAILABLE
        set_gpu_mode(use_gpu)
        
        # Determine processing mode
        is_prores = processing_mode == 'prores_proxy'
        use_nvenc = (processing_mode in ['h264_nvenc', 'hevc_nvenc']) and NVENC_AVAILABLE
        gpu_encoder = processing_mode if use_nvenc else 'none'
        
        python_str = "Portable" if USING_PORTABLE_PYTHON else "System"
        cuda_str = "CuPy CTK" if USING_CUPY_CTK else ("Portable" if USING_PORTABLE_CUDA else "System/None")

        # Determine FPS
        if custom_fps is not None and custom_fps > 0:
            output_fps = custom_fps
        else:
            output_fps = get_video_fps(local_video_paths[0])
            
        # Prepare output paths
        output_folder = get_output_dir()
        os.makedirs(output_folder, exist_ok=True)
        name, _ = os.path.splitext(output_filename)
        ext = '.mov' if is_prores else '.mp4'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
        output_path = os.path.join(output_folder, filename)
        temp_output = os.path.join(session_dir, filename)

        # ── Beat analysis (skipped on regenerate) ────────────────────
        # On regenerate, reuse cached beat_info from the previous render
        # so the user can tweak presets/effects/lyrics without waiting for
        # the expensive audio+video analysis to re-run.
        cached_beat_info = session_state.get('cached_beat_info')
        # Invalidate the cache if the Qwen semantic setting changed since the
        # last analysis — the tag set must match the user's choice.
        _qwen_changed = session_state.get('cached_qwen_enabled') != qwen_semantics
        _skip_analysis = False
        if regenerate and cached_beat_info and not _qwen_changed:
            selected_beats = session_state['cached_selected_beats']
            beat_info = cached_beat_info
            beat_times = beat_info.get('times', selected_beats)
            if console_logger:
                console_logger.line("🔄 Regenerating — reusing cached beat analysis")
            if progress_callback:
                progress_callback(_stage_status(6))
            _skip_analysis = True
        if not _skip_analysis:
            # Disk-level beat-analysis cache so an interrupted render can skip
            # stages 1-5 after an app restart (memory-limit crash on Render.com).
            _beat_key = None
            _disk_hit = False
            if _jobs is not None:
                _beat_key = _jobs.beat_cache_key(local_audio_path, local_video_paths, qwen_semantics)
                _hit = _jobs.load_beat_cache(_beat_key)
                if _hit is not None:
                    selected_beats, beat_info = _hit
                    beat_times = beat_info.get('times', selected_beats)
                    if console_logger:
                        console_logger.line("♻️  Beat analysis loaded from disk cache (stages 1-5 skipped)")
                    session_state['cached_beat_info'] = beat_info
                    session_state['cached_selected_beats'] = selected_beats
                    session_state['cached_qwen_enabled'] = qwen_semantics
                    if progress_callback:
                        progress_callback(_stage_status(6))
                    _stage5_summary(console_logger, beat_info.get("video_analysis"))
                    if job_id and _jobs is not None:
                        _jobs.save_job_beat_cache(job_id, selected_beats, beat_info)
                    _disk_hit = True

            if not _disk_hit:
                if console_logger and not qwen_semantics:
                    console_logger.line("🧠 Qwen semantic detection disabled — using legacy planner")
                selected_beats, beat_info = analyze_beats_auto(
                    local_audio_path,
                    use_gpu=use_gpu,
                    video_files=local_video_paths,
                    enable_qwen_semantics=qwen_semantics,
                    progress_callback=progress_callback,
                    console_callback=lambda stage, message: console_logger.stage_line(stage, message) if console_logger else None,
                )
                beat_times = beat_info.get('times', selected_beats)
                # Cache for future regeneration
                session_state['cached_beat_info'] = beat_info
                session_state['cached_selected_beats'] = selected_beats
                session_state['cached_qwen_enabled'] = qwen_semantics
                _stage5_summary(console_logger, beat_info.get("video_analysis"))
                # Persist to disk so a restart can resume.
                if _beat_key and _jobs is not None:
                    _jobs.save_beat_cache(_beat_key, selected_beats, beat_info)
                if job_id and _jobs is not None:
                    _jobs.save_job_beat_cache(job_id, selected_beats, beat_info)

                if progress_callback:
                    progress_callback(_stage_status(6))

        # Create video
        result_path = create_music_video(
            local_audio_path, local_video_paths, selected_beats,
            output_file=temp_output, max_workers=parallel_workers,
            beat_info=beat_info, lossless_mode=is_prores,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, fps=output_fps,
            effects_config=effects_config,
            smart_sync=smart_sync,
            transition_preset_name=transition_preset_name,
            job_id=job_id,
        )

        # ── Lyrics overlay ─────────────────────────────────────────────
        # If lyrics text was provided, generate an ASS subtitle file and
        # burn it into the rendered video with the selected font combo.
        if lyrics_text and lyrics_text.strip():
            try:
                from lyrics_engine import process_lyrics, burn_lyrics_into_video
                from font_presets import FONTS_DIR as _fd
                from ffmpeg_processing import get_video_resolution

                audio_dur = beat_info.get('audio_duration', 180.0)
                vid_w, vid_h = get_video_resolution(result_path)

                console_logger.line(f"🎤 Processing lyrics: {len(lyrics_text)} chars")

                ass_path, _lines, _combo = process_lyrics(
                    lyrics_text=lyrics_text,
                    audio_duration=audio_dur,
                    beat_times=beat_times,
                    font_combo_name=font_combo_name,
                    video_width=vid_w,
                    video_height=vid_h,
                    fps=output_fps,
                    output_dir=session_dir,
                )

                if ass_path and os.path.exists(ass_path):
                    lyrics_output = os.path.join(session_dir, f"lyrics_{timestamp}.mp4")
                    burn_lyrics_into_video(
                        video_file=result_path,
                        ass_file=ass_path,
                        output_file=lyrics_output,
                        fonts_dir=_fd,
                        use_nvenc=use_nvenc and NVENC_AVAILABLE,
                        gpu_encoder=gpu_encoder,
                        fps=output_fps,
                    )
                    # Replace the original output with the lyrics version
                    if os.path.exists(lyrics_output) and os.path.getsize(lyrics_output) > 1000:
                        try:
                            os.remove(result_path)
                        except OSError:
                            pass
                        result_path = lyrics_output
                        console_logger.line(f"✓ Lyrics overlay applied: {_combo.name}")
                    else:
                        console_logger.line("⚠️  Lyrics overlay failed; using video without lyrics")
            except Exception as lyrics_err:
                console_logger.line(f"⚠️  Lyrics processing skipped: {lyrics_err}")

        # Move to output folder
        shutil.move(result_path, output_path)

        # Compress the finalized video to fit within a dynamic size cap.
        # The cap is 2x the total size of all uploaded source files
        # (audio + videos). An explicit BEATSYNC_MAX_OUTPUT_MB env var, if set,
        # overrides this as a hard ceiling. Skip when already under the cap.
        _bsync_input_bytes = os.path.getsize(local_audio_path)
        for _vp in local_video_paths:
            _bsync_input_bytes += os.path.getsize(_vp)
        _bsync_dynamic_mb = (_bsync_input_bytes * 2) / (1024 * 1024)
        _bsync_env_cap = os.environ.get('BEATSYNC_MAX_OUTPUT_MB', '').strip()
        if _bsync_env_cap:
            _bsync_max_mb = min(int(_bsync_env_cap), int(_bsync_dynamic_mb)) if _bsync_dynamic_mb else int(_bsync_env_cap)
        else:
            _bsync_max_mb = max(50, int(_bsync_dynamic_mb))
        console_logger.line(
            f"Output size cap: {_bsync_max_mb} MB "
            f"(2x uploaded input: {round(_bsync_input_bytes / 1048576, 1)} MB)"
        )
        if os.path.getsize(output_path) > _bsync_max_mb * 1024 * 1024:
            console_logger.line(f"Output exceeds {_bsync_max_mb} MB; compressing...")
            compressed_ext = '.mp4'
            compressed_name = f"{name}_{timestamp}{compressed_ext}"
            compressed_path = os.path.join(session_dir, compressed_name)
            compress_video_to_target_size(
                input_file=output_path,
                output_file=compressed_path,
                target_size_mb=_bsync_max_mb,
                use_nvenc=use_nvenc and NVENC_AVAILABLE,
                gpu_encoder=gpu_encoder,
            )
            # Replace the oversized file with the compressed H.264 copy.
            try:
                os.remove(output_path)
            except OSError:
                pass
            final_path = os.path.join(output_folder, compressed_name)
            shutil.move(compressed_path, final_path)
            output_path = final_path
            filename = compressed_name
            # ProRes mode is now effectively an H.264 playback copy.
            is_prores = False

        # Create preview for ProRes if needed
        preview_path = output_path
        if is_prores:
            preview_filename = f"{name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(session_dir, preview_filename)
            preview_cmd = [FFMPEG_PATH]
            if NVENC_AVAILABLE:
                preview_cmd.extend(['-hwaccel', 'cuda', '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '23'])
            else:
                preview_cmd.extend(['-hwaccel', 'auto', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23'])
            preview_cmd.extend(['-i', output_path, '-pix_fmt', 'yuv420p', '-y', preview_path])
            subprocess.run(preview_cmd, capture_output=True, text=True, timeout=180)

        # ── Web optimisation pass ────────────────────────────────────
        # Always produce a browser-ready MP4 so the output downloads fast
        # and plays instantly in the Gradio <video> component:
        #   • H.264 High + AAC in MP4
        #   • moov atom at front (+faststart → progressive download)
        #   • 2-second GOP for smooth seeking
        #   • scaled to ≤1920px wide (keeps full HD, shrinks 4K sources)
        #
        # For ProRes mode we optimise the preview; for H.264 mode we
        # optimise the final output directly.  The optimised file replaces
        # what is served to the user.
        web_src = preview_path if is_prores else output_path
        if not web_src.lower().endswith('.mp4') or is_prores:
            web_name = f"{name}_{timestamp}_web.mp4"
            web_path = os.path.join(session_dir, web_name)
            try:
                optimize_for_web(
                    input_file=web_src,
                    output_file=web_path,
                    fps=output_fps,
                    use_nvenc=use_nvenc and NVENC_AVAILABLE,
                    gpu_encoder=gpu_encoder,
                )
                # Move the web-optimised file to the output folder and use it
                final_web = os.path.join(output_folder, web_name)
                shutil.move(web_path, final_web)
                preview_path = final_web
                # For non-ProRes, replace the main output too
                if not is_prores:
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                    output_path = final_web
                    filename = web_name
                console_logger.line(
                    f"🌐 Web-optimised: {os.path.getsize(final_web) / 1048576:.1f} MB "
                    f"(faststart + AAC + 2s GOP)"
                )
            except Exception as web_err:
                console_logger.line(f"⚠️  Web optimisation skipped: {web_err}")
        _stage6_summary(console_logger, beat_info)

        # Generate status message based on mode
        gpu_info = f"⚡ GPU: {GPU_INFO}" if use_gpu else "💻 CPU"
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"
        audio_info = "AAC 192k (48kHz) — web-optimised"
        
        if is_prores:
            codec_info = "ProRes 422 Proxy (.mov) - Lossless"
            encoder_info = "🎯 Lossless Concatenation"
        elif use_nvenc:
            codec_info = f"{gpu_encoder.upper()} (.mp4)"
            encoder_info = f"⚡ {gpu_encoder.upper()}"
        else:
            codec_info = "H.264 (.mp4)"
            encoder_info = "💻 libx264"

        total_cuts = len(selected_beats) - 1
        sections_info = beat_info.get('selection_info', [])
        total_processing_seconds = time.perf_counter() - total_started
        processing_label = gpu_encoder.upper() if use_nvenc else ("PRORES_PROXY" if is_prores else "H264_CPU")
        
        status_msg = get_success_message_auto(
            total_cuts, len(beat_times),
            beat_info.get('tempo', 120), sections_info,
            python_str, cuda_str, MAX_THREADS, CPU_COUNT,
            parallel_workers, gpu_info, encoder_info,
            codec_info, fps_info, filename, audio_info,
            audio_duration=beat_info.get('audio_duration'),
            output_fps=output_fps,
            total_processing_seconds=total_processing_seconds,
            processing_label=processing_label
        )
        # Return preview path for display + download, keep session_state intact
        # Validate final output — guard against 0-byte / missing files
        # so the UI never serves a broken video.
        if not preview_path or not os.path.exists(preview_path) or os.path.getsize(preview_path) < 1000:
            console_logger.line("⚠️  Final output missing or too small; render may have failed.")
            if job_id and _jobs is not None:
                _jobs.mark_failed(job_id, "no valid output")
            return None, "❌ Error: Render produced no valid output. Check console for details.", session_state, gr.update(visible=False)

        if job_id and _jobs is not None:
            _jobs.set_output(job_id, preview_path)
        return preview_path, status_msg, session_state, gr.update(visible=True, value=preview_path)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        import traceback
        traceback.print_exc()
        if job_id and _jobs is not None:
            _jobs.mark_failed(job_id, str(e))
        return None, error_msg, session_state, gr.update(visible=False)


def process_video(audio_file: str, video_files: VideoFilesInput,
                 output_filename: str, processing_mode: str,
                 custom_fps: float, session_state: dict,
                 zip_file: str | None = None,
                 effects_enabled: bool = False,
                 gradient_overlay: bool = False,
                 vignette: bool = False,
                 zoom_punch: bool = False,
                 shake: bool = False,
                 flash_on_beat: bool = False,
                 color_boost: bool = False,
                 glitch: bool = False,
                 mirror: bool = False,
                 slow_motion: bool = False,
                 slow_motion_factor: float = 2.0,
                 transition_type: str = 'none',
                 transition_duration: float = 0.35,
                 vignette_strength: float = 0.4,
                 zoom_punch_strength: float = 1.06,
                 shake_strength: float = 8.0,
                 flash_intensity: float = 0.6,
                 color_boost_amount: float = 1.3,
                 glitch_strength: float = 0.3,
                 slow_motion_probability: float = 0.15,
                 smart_sync: bool = False,
                 lyrics_text: str = "",
                 font_combo_name: str = "",
                 transition_preset_name: str = "",
                 qwen_semantics: bool = True,
                 regenerate: bool = False,
                 resume_job_id: str | None = None,
                 ) -> Iterator[StatusResult]:
    status_queue: queue.Queue[str | None] = queue.Queue()
    result_queue: queue.Queue[StatusResult] = queue.Queue(maxsize=1)
    log_lines: list[str] = []
    render_started = time.perf_counter()

    # ── Persistent render job ───────────────────────────────────────
    # A job records inputs/settings on disk so a render can resume after an
    # app restart (e.g. a Render.com container that exceeded its memory limit).
    job_id = None
    if _jobs is not None:
        if resume_job_id and _jobs.load_record(resume_job_id):
            # Resuming an interrupted job — reuse its persisted inputs + seed.
            job_id = resume_job_id
            rec = _jobs.load_record(job_id) or {}
            if rec.get("inputs"):
                audio_file = rec["inputs"].get("audio_file") or audio_file
                video_files = rec["inputs"].get("video_files") or video_files
                zip_file = rec["inputs"].get("zip_file") or zip_file
            if rec.get("seed"):
                session_state["_resume_seed"] = rec["seed"]
            _jobs.update_job(job_id, status=_jobs.STATUS_RUNNING, stage="resuming")
            log_lines.append(f"♻️  Resuming job {job_id} after restart")
        elif not regenerate:
            # Fresh render — copy uploads into a durable job folder.
            import random as _seed_rng
            seed = int(_seed_rng.random() * (2 ** 31))
            session_state["_resume_seed"] = seed
            job_id = _jobs.create_job({
                "seed": seed,
                "output_filename": output_filename,
                "processing_mode": processing_mode,
                "qwen_semantics": bool(qwen_semantics and QWEN_BACKEND_AVAILABLE),
                "smart_sync": smart_sync,
            })
            try:
                persisted = _jobs.persist_inputs(job_id, audio_file, video_files or [], zip_file)
                _jobs.update_job(job_id, inputs=persisted)
            except Exception:
                pass
            _jobs.trim_old_jobs()

    def push_line(line: str) -> None:
        log_lines.append(line)
        status_queue.put(line)
        if job_id and _jobs is not None:
            _jobs.append_log(job_id, line)
            m = re.search(r"Progress: (\d+)/(\d+) clips", line)
            if m:
                _jobs.update_job(job_id, stage=f"rendering clips {m.group(1)}/{m.group(2)}")

    log_capture = LiveLogCapture(sys.__stdout__, push_line)
    console_logger = StageConsoleLogger(log_capture)

    # Resolve "Random" selections — pick a random font combo and/or
    # transition preset so each render produces different results.
    if not font_combo_name:
        import random as _rng
        from font_presets import get_random_combo as _grc
        font_combo_name = _grc(_rng.Random()).name
    if not transition_preset_name:
        import random as _rng2
        from transition_presets import get_random_preset as _grp
        transition_preset_name = _grp(_rng2.Random()).name

    def progress_callback(message: str) -> None:
        status_queue.put(message)
        if job_id and _jobs is not None:
            _jobs.update_job(job_id, stage=message[:120])
        match = re.search(r"Stage (\d+) is processing", message)
        if match:
            console_logger.start_stage(int(match.group(1)))

    def worker() -> None:
        try:
            with contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                effects_config = parse_effects_from_ui(
                    enabled=effects_enabled,
                    gradient_overlay=gradient_overlay,
                    vignette=vignette,
                    zoom_punch=zoom_punch,
                    shake=shake,
                    flash_on_beat=flash_on_beat,
                    color_boost=color_boost,
                    glitch=glitch,
                    mirror=mirror,
                    slow_motion=slow_motion,
                    slow_motion_factor=slow_motion_factor,
                    transition_type=transition_type,
                    transition_duration=transition_duration,
                    vignette_strength=vignette_strength,
                    zoom_punch_strength=zoom_punch_strength,
                    shake_strength=shake_strength,
                    flash_intensity=flash_intensity,
                    color_boost_amount=color_boost_amount,
                    glitch_strength=glitch_strength,
                    slow_motion_probability=slow_motion_probability,
                )
                result = _process_video_impl(
                    audio_file=audio_file,
                    video_files=video_files,
                    output_filename=output_filename,
                    processing_mode=processing_mode,
                    custom_fps=custom_fps,
                    session_state=session_state,
                    effects_config=effects_config,
                    smart_sync=smart_sync,
                    lyrics_text=lyrics_text,
                    font_combo_name=font_combo_name,
                    transition_preset_name=transition_preset_name,
                    progress_callback=progress_callback,
                    console_logger=console_logger,
                    regenerate=regenerate,
                    zip_file=zip_file,
                    qwen_semantics=bool(qwen_semantics and QWEN_BACKEND_AVAILABLE),
                    job_id=job_id,
                )
        except Exception as e:
            console_logger.line(f"Error: {e}")
            result = None, f"❌ Error: {e}", session_state, gr.update(visible=False)
        finally:
            console_logger.finish()
        result_queue.put(result)
        status_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    console_logger.start_stage(1)
    thread.start()

    max_log_lines = 60

    def render_feed() -> str:
        elapsed = time.perf_counter() - render_started
        header = f"⏱️ {elapsed:0.1f}s elapsed  |  {len(log_lines)} log lines"
        if job_id:
            header += f"  |  job {job_id}"
        tail = log_lines[-max_log_lines:]
        return header + "\n" + "\n".join(tail)

    yield None, render_feed(), session_state, gr.update(visible=False)

    while True:
        message = status_queue.get()
        if message is None:
            break
        yield None, render_feed(), session_state, gr.update(visible=False)

    thread.join()
    # If the worker thread dies without producing a result (e.g. OOM kill of
    # the subprocess), mark the job interrupted so it can be resumed later.
    if job_id and _jobs is not None:
        rec = _jobs.load_record(job_id)
        if rec and rec.get("status") == _jobs.STATUS_RUNNING:
            _jobs.mark_interrupted(job_id)
    yield result_queue.get()


def regenerate_video(audio_file: str, video_files: VideoFilesInput,
                      output_filename: str, processing_mode: str,
                      custom_fps: float, session_state: dict,
                      zip_file: str | None = None,
                      effects_enabled: bool = False,
                      gradient_overlay: bool = False,
                      vignette: bool = False,
                      zoom_punch: bool = False,
                      shake: bool = False,
                      flash_on_beat: bool = False,
                      color_boost: bool = False,
                      glitch: bool = False,
                      mirror: bool = False,
                      slow_motion: bool = False,
                      slow_motion_factor: float = 2.0,
                      transition_type: str = 'none',
                      transition_duration: float = 0.35,
                      vignette_strength: float = 0.4,
                      zoom_punch_strength: float = 1.06,
                      shake_strength: float = 8.0,
                      flash_intensity: float = 0.6,
                      color_boost_amount: float = 1.3,
                      glitch_strength: float = 0.3,
                      slow_motion_probability: float = 0.15,
                      smart_sync: bool = False,
                      lyrics_text: str = "",
                      font_combo_name: str = "",
                      transition_preset_name: str = "",
                      qwen_semantics: bool = True,
                      ) -> Iterator[StatusResult]:
    """Re-render with updated settings, reusing cached beat analysis.

    Falls back to a full render if no cached beat_info exists.
    """
    if not session_state or 'cached_beat_info' not in session_state:
        yield from process_video(
            audio_file, video_files, output_filename, processing_mode,
            custom_fps, session_state, zip_file,
            effects_enabled, gradient_overlay, vignette, zoom_punch, shake,
            flash_on_beat, color_boost, glitch, mirror,
            slow_motion, slow_motion_factor,
            transition_type, transition_duration,
            vignette_strength, zoom_punch_strength,
            shake_strength, flash_intensity,
            color_boost_amount, glitch_strength,
            slow_motion_probability,
            smart_sync, lyrics_text, font_combo_name,
            transition_preset_name, qwen_semantics,
        )
        return

    yield from process_video(
        audio_file, video_files, output_filename, processing_mode,
        custom_fps, session_state, zip_file,
        effects_enabled, gradient_overlay, vignette, zoom_punch, shake,
        flash_on_beat, color_boost, glitch, mirror,
        slow_motion, slow_motion_factor,
        transition_type, transition_duration,
        vignette_strength, zoom_punch_strength,
        shake_strength, flash_intensity,
        color_boost_amount, glitch_strength,
        slow_motion_probability,
        smart_sync, lyrics_text, font_combo_name,
        transition_preset_name, qwen_semantics,
        regenerate=True,
    )


def cleanup_on_startup():
    """
    Clean temporary runtime files on script start while preserving user inputs
    and the persistent video analysis cache.
    """
    input_base = get_input_dir()
    protected_dirs = {'audio', 'video', 'video_analysis_cache', 'render_jobs', 'beat_analysis_cache'}

    try:
        os.makedirs(get_audio_input_dir(), exist_ok=True)
        os.makedirs(get_video_input_dir(), exist_ok=True)
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)

        if os.path.exists(input_base):
            for item in os.listdir(input_base):
                item_path = os.path.join(input_base, item)

                # Keep the latest user input files across restarts.
                if item in protected_dirs:
                    continue

                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                except Exception as e:
                    print(f"   ⚠️  Could not clean {item}: {e}")

        # Recreate runtime temp upload folder after cleanup.
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)

    except Exception as e:
        print(f"   ⚠️  Warning during startup cleanup: {e}")


def create_ui() -> gr.Blocks:
    # These definitions are needed within the function's scope
    python_status = "✅ Portable (bin/python-3.13.14-embed-amd64/)" if USING_PORTABLE_PYTHON else "⚠️  System Python"
    if USING_CUPY_CTK:
        cuda_status = "✅ CuPy CTK (Python wheel libraries)"
    elif USING_PORTABLE_CUDA:
        cuda_status = "✅ Portable (bin/CUDA/v13.3)"
    else:
        cuda_status = "⚠️  System CUDA (or not available)"
    ffmpeg_status = "✅ Portable (bin/ffmpeg/)" if FFMPEG_FOUND else "⚠️  System FFmpeg"
    
    app = gr.Blocks(title='BeatSync Engine', theme='ocean', css=STATUS_BOX_CSS)
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### 📁 Input Files')
                audio_input = gr.File(label=LABEL_AUDIO_FILE, file_types=['.mp3', '.wav', '.flac'], type='filepath', elem_id='audio-file-input')
                video_input = gr.File(label=LABEL_VIDEO_FILES, file_count='multiple', file_types=['.mp4', '.mkv'], type='filepath', elem_id='video-files-input')
                zip_input = gr.File(label=LABEL_ZIP_FILE, file_types=['.zip'], type='filepath', elem_id='zip-clips-input')

                with gr.Group():
                    gr.Markdown('### ⚙️ Video Settings')
                    custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)

                with gr.Group():
                    gr.Markdown(f'### 🎬 Processing Mode')
                    if NVENC_AVAILABLE:
                        processing_mode = gr.Radio(choices=[('NVIDIA NVENC H.264', 'h264_nvenc'), ('NVIDIA NVENC HEVC (H.265)', 'hevc_nvenc'), ('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='h264_nvenc', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_nvenc())
                    else:
                        processing_mode = gr.Radio(choices=[('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='cpu', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_cpu())
                
                with gr.Group():
                    gr.Markdown('### 📁 Output Settings')
                    output_filename = gr.Textbox(value='music_video.mp4', label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)

                with gr.Accordion('✨ Visual Effects & Transitions', open=False):
                    effects_enabled = gr.Checkbox(value=False, label=LABEL_EFFECTS_ENABLED, info=INFO_EFFECTS_ENABLED)
                    smart_sync_enabled = gr.Checkbox(value=False, label=LABEL_SMART_SYNC, info=INFO_SMART_SYNC)
                    qwen_semantics_enabled = gr.Checkbox(
                        value=QWEN_BACKEND_AVAILABLE,
                        interactive=QWEN_BACKEND_AVAILABLE,
                        label=(LABEL_QWEN_SEMANTICS if QWEN_BACKEND_AVAILABLE
                               else LABEL_QWEN_SEMANTICS + ' — unavailable (model not found)'),
                        info=INFO_QWEN_SEMANTICS,
                        elem_id='qwen-semantics-toggle',
                    )
                    with gr.Row():
                        gradient_overlay = gr.Checkbox(value=False, label=LABEL_GRADIENT_OVERLAY, info=INFO_GRADIENT_OVERLAY)
                        vignette = gr.Checkbox(value=False, label=LABEL_VIGNETTE, info=INFO_VIGNETTE)
                    with gr.Row():
                        zoom_punch = gr.Checkbox(value=False, label=LABEL_ZOOM_PUNCH, info=INFO_ZOOM_PUNCH)
                        shake = gr.Checkbox(value=False, label=LABEL_SHAKE, info=INFO_SHAKE)
                    with gr.Row():
                        flash_on_beat = gr.Checkbox(value=False, label=LABEL_FLASH_BEAT, info=INFO_FLASH_BEAT)
                        color_boost = gr.Checkbox(value=False, label=LABEL_COLOR_BOOST, info=INFO_COLOR_BOOST)
                    with gr.Row():
                        glitch = gr.Checkbox(value=False, label=LABEL_GLITCH, info=INFO_GLITCH)
                        mirror = gr.Checkbox(value=False, label=LABEL_MIRROR, info=INFO_MIRROR)
                    with gr.Row():
                        slow_motion = gr.Checkbox(value=False, label=LABEL_SLOW_MOTION, info=INFO_SLOW_MOTION)
                        slow_motion_factor = gr.Number(value=2.0, minimum=1.25, maximum=4.0, step=0.25, label=LABEL_SLOW_MOTION_FACTOR, info=INFO_SLOW_MOTION_FACTOR)
                    with gr.Row():
                        transition_type = gr.Dropdown(
                            choices=[(label, key) for key, label in TRANSITION_TYPES.items()],
                            value='none',
                            label=LABEL_TRANSITION_TYPE,
                            info=INFO_TRANSITION_TYPE,
                        )
                        transition_duration = gr.Slider(0.1, 1.0, value=0.35, step=0.05, label=LABEL_TRANSITION_DURATION, info=INFO_TRANSITION_DURATION)
                    with gr.Accordion('🎚️ Fine-tune Effect Strengths', open=False):
                        with gr.Row():
                            vignette_strength = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label=LABEL_VIGNETTE_STRENGTH)
                            zoom_punch_strength = gr.Slider(1.01, 1.5, value=1.06, step=0.01, label=LABEL_ZOOM_PUNCH_STRENGTH)
                        with gr.Row():
                            shake_strength = gr.Slider(2, 40, value=8, step=1, label=LABEL_SHAKE_STRENGTH)
                            flash_intensity = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label=LABEL_FLASH_INTENSITY)
                        with gr.Row():
                            color_boost_amount = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label=LABEL_COLOR_BOOST_AMOUNT)
                            glitch_strength = gr.Slider(0.05, 1.0, value=0.3, step=0.05, label=LABEL_GLITCH_STRENGTH)
                        slow_motion_probability = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label=LABEL_SLOW_MOTION_PROBABILITY)

                with gr.Accordion(LABEL_LYRICS_SECTION, open=False):
                    lyrics_text = gr.Textbox(
                        label=LABEL_LYRICS_TEXT,
                        info=INFO_LYRICS_TEXT,
                        placeholder="Paste song lyrics here...\n\nSupports:\n• Plain text (auto-distributed across song)\n• LRC format: [00:05.00] First line here\n• Any language → auto-transliterated to English",
                        lines=8,
                        max_lines=20,
                    )
                    with gr.Row():
                        font_combo = gr.Dropdown(
                            choices=_FONT_COMBO_CHOICES,
                            value="",
                            label=LABEL_FONT_COMBO,
                            info=INFO_FONT_COMBO,
                        )
                        transition_preset = gr.Dropdown(
                            choices=_TRANSITION_PRESET_CHOICES,
                            value="",
                            label=LABEL_TRANSITION_PRESET,
                            info=INFO_TRANSITION_PRESET,
                        )

                with gr.Row():
                    process_btn = gr.Button('🎬 Create Music Video', variant='primary', size='lg')
                    regenerate_btn = gr.Button('🔄 Regenerate', variant='secondary', size='lg', interactive=True)

            with gr.Column(scale=1):
                gr.Markdown('### 📺 Output')
                with gr.Group(visible=False) as resume_group:
                    gr.Markdown(f"### {LABEL_RESUME_SECTION}")
                    gr.Markdown(INFO_RESUME_SECTION)
                    resume_status = gr.Markdown('_No interrupted render to resume yet._')
                    resume_btn = gr.Button('♻️ Resume Last Render', variant='secondary')
                status_output = gr.Textbox(label='Live Processing Log', interactive=False, value=get_ready_status(python_status, cuda_status, MAX_THREADS, CPU_COUNT, ffmpeg_status, GPU_AVAILABLE, gpu_info, NVENC_AVAILABLE), lines=20, max_lines=20, elem_id='status-output-box', autoscroll=True)
                video_output = gr.Video(label='Generated Music Video', format='mp4', interactive=False, elem_id='generated-video-output')
                download_btn = gr.DownloadButton('⬇️ Download MP4', variant='secondary', visible=False, elem_id='download-video-btn')

        _shared_inputs = [
            audio_input, video_input,
            output_filename, processing_mode, custom_fps,
            session_state,
            zip_input,
            effects_enabled,
            gradient_overlay, vignette, zoom_punch, shake,
            flash_on_beat, color_boost, glitch, mirror,
            slow_motion, slow_motion_factor,
            transition_type, transition_duration,
            vignette_strength, zoom_punch_strength,
            shake_strength, flash_intensity,
            color_boost_amount, glitch_strength,
            slow_motion_probability,
            smart_sync_enabled,
            lyrics_text,
            font_combo,
            transition_preset,
            qwen_semantics_enabled,
        ]
        _shared_outputs = [video_output, status_output, session_state, download_btn]

        process_btn.click(
            fn=process_video,
            inputs=_shared_inputs,
            outputs=_shared_outputs,
            show_progress='hidden'
        ).then(
            fn=None,
            js=_AUTO_DOWNLOAD_JS,
        )

        regenerate_btn.click(
            fn=regenerate_video,
            inputs=_shared_inputs,
            outputs=_shared_outputs,
            show_progress='hidden'
        ).then(
            fn=None,
            js=_AUTO_DOWNLOAD_JS,
        )

        # ── Resume / refresh-persistence wiring ─────────────────────────
        # On page load: restore settings from localStorage and reveal the
        # Resume panel when an interrupted job exists on disk.
        app.load(
            fn=_load_resume_panel,
            inputs=None,
            outputs=[resume_group, resume_status],
            show_progress='hidden',
        ).then(fn=None, js=_SETTINGS_PERSISTENCE_JS)

        resume_btn.click(
            fn=_resume_last_job,
            inputs=None,
            outputs=_shared_outputs,
            show_progress='hidden',
        ).then(
            fn=None,
            js=_AUTO_DOWNLOAD_JS,
        )

    return app

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Clean up old files only on startup
    cleanup_on_startup()
    
    app = create_ui()
    launch_port = find_launch_port()
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=launch_port,
        share=False,
        inbrowser=bool(os.environ.get("GRADIO_INBROWSER", "")),
        show_error=True
    )
