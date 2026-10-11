"""Locate (and, if necessary, install) the FFmpeg binaries.

The app needs both ``ffmpeg`` and ``ffprobe``. A Codespace built from the
default image ships ffmpeg in the system image, but the system image can be
rebuilt without it — leaving the app with ``No such file or directory:
'ffmpeg'`` mid-render. To stay usable regardless of the base image, this
module resolves the binaries in priority order and, on Linux with apt
available, installs ffmpeg on demand:

  1. ``BEATSYNC_FFMPEG_PATH`` (and ``BEATSYNC_FFPROBE_PATH``) env override
  2. ``<repo>/bin/ffmpeg/ffmpeg`` portable bundle (gitignored, optional)
  3. ``shutil.which`` on PATH
  4. apt-get install (Debian/Ubuntu) — best effort, never fatal

Results are cached for the process lifetime.
"""
import os
import shutil
import subprocess
import sys
from typing import Optional, Tuple

_CACHE: Optional[Tuple[str, str]] = None
_APT_ATTEMPTED = False

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PORTABLE_DIR = os.path.join(_ROOT_DIR, 'bin', 'ffmpeg')


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == 'nt' else name


def _portable(name: str) -> Optional[str]:
    candidate = os.path.join(_PORTABLE_DIR, _exe(name))
    return candidate if os.path.isfile(candidate) else None


def _on_path(name: str) -> Optional[str]:
    return shutil.which(name)


def _install_ffmpeg_via_apt() -> bool:
    """Best-effort apt install of ffmpeg. Returns True if it now resolves."""
    global _APT_ATTEMPTED
    if _APT_ATTEMPTED or os.name == 'nt':
        return False
    _APT_ATTEMPTED = True
    if not shutil.which('apt-get'):
        return False

    def _run(cmd: list[str]) -> bool:
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
            return True
        except Exception:
            return False

    # `sudo -n` succeeds only with passwordless sudo; fall back to a direct
    # call for containers that already run as root.
    sudo = ['sudo', '-n'] if shutil.which('sudo') else []
    if sudo and not _run(sudo + ['true']):
        sudo = []

    print("⚠️  ffmpeg not found — attempting to install it (apt-get)…")
    _run(sudo + ['apt-get', 'update', '-qq'])
    env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
    try:
        subprocess.run(
            sudo + ['apt-get', 'install', '-y', '--no-install-recommends', 'ffmpeg'],
            capture_output=True, text=True, timeout=900, check=False, env=env,
        )
    except Exception:
        pass

    return bool(shutil.which('ffmpeg'))


def resolve_ffmpeg() -> Tuple[str, str]:
    """Return ``(ffmpeg, ffprobe)`` executable paths (or bare names)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    ffmpeg = os.environ.get('BEATSYNC_FFMPEG_PATH') or None
    ffprobe = os.environ.get('BEATSYNC_FFPROBE_PATH') or None

    if not ffmpeg:
        ffmpeg = _portable('ffmpeg') or _on_path('ffmpeg')
    if not ffprobe:
        ffprobe = _portable('ffprobe') or _on_path('ffprobe')

    if not ffmpeg and _install_ffmpeg_via_apt():
        ffmpeg = _on_path('ffmpeg')
        ffprobe = ffprobe or _on_path('ffprobe')

    if not ffprobe and ffmpeg:
        # Derive ffprobe from the ffmpeg location when possible.
        sibling = os.path.join(os.path.dirname(os.path.abspath(ffmpeg)), _exe('ffprobe'))
        ffprobe = sibling if os.path.isfile(sibling) else _on_path('ffprobe')

    _CACHE = (ffmpeg or 'ffmpeg', ffprobe or 'ffprobe')
    if ffmpeg:
        print(f"   FFmpeg binary: {ffmpeg}")
    else:
        print("   ⚠️  FFmpeg could not be located; renders will fail until it is installed.",
              file=sys.stderr)
    return _CACHE
