"""Shot-boundary detection and media probing.

Reconstructs the cut list from a finished video so the edit timeline can be
rebuilt (approximately). Uses PySceneDetect when it is installed, otherwise
falls back to ffmpeg's scene-change filter — no extra dependency required.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

try:  # package-relative when imported as ``forensics.scenes``
    from ..media_paths import resolve_ffmpeg
except ImportError:  # ``src`` on sys.path (the app's flat import style)
    from media_paths import resolve_ffmpeg

_SHOWINFO_PTS = re.compile(r"pts_time:([0-9.]+)")
_SCENE_SCORE = re.compile(r"scene_score=([0-9.eE+-]+)")


@dataclass
class MediaProbe:
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    nb_frames: int = 0
    bit_rate: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout or "") + (proc.stderr or "")


def probe_media(path: str, ffprobe: Optional[str] = None) -> MediaProbe:
    """Read duration, fps, resolution and codec via ffprobe."""
    _, ffprobe_exe = resolve_ffmpeg()
    ffprobe = ffprobe or ffprobe_exe
    out = _run([ffprobe, "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", path])
    probe = MediaProbe()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return probe

    fmt = data.get("format", {})
    probe.duration = float(fmt.get("duration") or 0.0)
    probe.bit_rate = int(float(fmt.get("bit_rate") or 0))

    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        probe.codec = stream.get("codec_name", "")
        probe.width = int(stream.get("width") or 0)
        probe.height = int(stream.get("height") or 0)
        probe.nb_frames = int(stream.get("nb_frames") or 0)
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        try:
            num, den = rate.split("/")
            probe.fps = float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            probe.fps = 0.0
        break
    return probe


@dataclass
class SceneResult:
    method: str = ""
    threshold: float = 0.0
    cut_times: List[float] = field(default_factory=list)
    scenes: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "cut_count": len(self.cut_times),
            "cut_times": [round(t, 3) for t in self.cut_times],
            "scenes": self.scenes,
        }


def _detect_with_pyscenedetect(path: str, threshold: float,
                               adaptive: bool = False) -> Optional[SceneResult]:
    """Content/adaptive detection via the optional ``scenedetect`` package."""
    try:
        from scenedetect import detect, ContentDetector, AdaptiveDetector  # type: ignore
    except Exception:
        return None
    try:
        detector = (AdaptiveDetector() if adaptive
                    else ContentDetector(threshold=threshold * 100.0))
        scene_list = detect(path, detector)
    except Exception:
        return None
    method = "pyscenedetect-adaptive" if adaptive else "pyscenedetect-content"
    res = SceneResult(method=method, threshold=threshold)
    for start, end in scene_list:
        res.scenes.append({"start": start.get_seconds(), "end": end.get_seconds()})
    res.cut_times = [s["start"] for s in res.scenes[1:]]
    return res


def _detect_with_ffmpeg(path: str, threshold: float) -> SceneResult:
    ffmpeg, _ = resolve_ffmpeg()
    vf = f"select='gt(scene,{threshold})',showinfo"
    out = _run([ffmpeg, "-v", "info", "-i", path, "-filter:v", vf,
                "-f", "null", "-"])
    res = SceneResult(method="ffmpeg-scene", threshold=threshold)
    for line in out.splitlines():
        if "pts_time:" not in line:
            continue
        m = _SHOWINFO_PTS.search(line)
        if not m:
            continue
        t = float(m.group(1))
        if t <= 0:
            continue
        score_m = _SCENE_SCORE.search(line)
        res.scenes.append({"start": t, "score": float(score_m.group(1)) if score_m else None})
    res.cut_times = [s["start"] for s in res.scenes]
    return res


def detect_scenes(path: str, threshold: float = 0.4,
                  method: str = "auto") -> SceneResult:
    """Detect shot boundaries.

    ``method`` is ``auto`` (PySceneDetect if installed, else ffmpeg),
    ``content`` / ``adaptive`` (force PySceneDetect), or ``ffmpeg``.
    """
    if method in ("content", "adaptive"):
        res = _detect_with_pyscenedetect(path, threshold,
                                         adaptive=(method == "adaptive"))
        if res is not None:
            return res
        method = "auto"
    if method == "ffmpeg":
        return _detect_with_ffmpeg(path, threshold)
    return (_detect_with_pyscenedetect(path, threshold)
            or _detect_with_ffmpeg(path, threshold))
