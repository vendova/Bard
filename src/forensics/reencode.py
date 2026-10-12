"""Reconstruct the GOP/keyframe timeline and flag re-encoding evidence.

Recovered evidence:

* **GOP structure** — keyframe index intervals reveal the encoder's GOP length.
  A single constant GOP across the whole file is typical of a straight
  transcode; irregular GOPs suggest the file was assembled from pieces.
* **Edit lists** — an ``elst`` means the writer trimmed/offset the media, the
  normal signature of a cut/trim performed after encoding.
* **Keyframe/splice look-ahead** — where a new GOP starts *exactly* at a cut,
  the encoder knew about the cut (an edit); where it does not, the encoder was
  encoding continuously and the cut was imposed afterwards (a re-encode/splice).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import List, Optional

from .containers import ContainerInfo


@dataclass
class GopStats:
    frame_count: int = 0
    keyframe_count: int = 0
    gop_lengths: List[int] = field(default_factory=list)

    @property
    def gop_mean(self) -> float:
        return mean(self.gop_lengths) if self.gop_lengths else 0.0

    @property
    def gop_std(self) -> float:
        return pstdev(self.gop_lengths) if len(self.gop_lengths) > 1 else 0.0

    @property
    def uniform(self) -> bool:
        """True when GOP length is (near) constant — a straight encode."""
        return bool(self.gop_lengths) and self.gop_std <= 0.5


@dataclass
class ReencodeEvidence:
    gop: GopStats = field(default_factory=GopStats)
    has_edit_list: bool = False
    # Cut timestamps (seconds) that are NOT preceded by a keyframe.
    misaligned_cuts: List[float] = field(default_factory=list)
    spliced_cuts: List[float] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    reencoded: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "gop": {
                "frame_count": self.gop.frame_count,
                "keyframe_count": self.gop.keyframe_count,
                "gop_mean": round(self.gop.gop_mean, 3),
                "gop_std": round(self.gop.gop_std, 3),
                "uniform": self.gop.uniform,
            },
            "has_edit_list": self.has_edit_list,
            "misaligned_cuts": [round(t, 3) for t in self.misaligned_cuts],
            "spliced_cuts": [round(t, 3) for t in self.spliced_cuts],
            "reencoded": self.reencoded,
            "findings": self.findings,
        }


def gop_stats(info: ContainerInfo, fps: float = 0.0) -> GopStats:
    """Derive GOP lengths (in frames) from the keyframe sample indices."""
    stats = GopStats(frame_count=info.sample_count,
                     keyframe_count=len(info.keyframe_indices))
    idxs = sorted(i for i in info.keyframe_indices if i >= 1)
    for prev, cur in zip(idxs, idxs[1:]):
        stats.gop_lengths.append(cur - prev)
    if info.sample_count and idxs:
        stats.gop_lengths.append(info.sample_count - idxs[-1] + 1)
    return stats


def _cut_frames(cut_times: List[float], fps: float) -> List[int]:
    if fps <= 0:
        return []
    return [max(1, int(round(t * fps)) + 1) for t in cut_times]


def analyze_reencode(
    info: ContainerInfo,
    cut_times: Optional[List[float]] = None,
    fps: float = 0.0,
) -> ReencodeEvidence:
    """Assemble re-encoding evidence, optionally cross-checking detected cuts."""
    ev = ReencodeEvidence()
    ev.gop = gop_stats(info, fps)
    ev.has_edit_list = info.has_edit_list

    if info.edit_list_is_trim:
        ev.findings.append(
            "Edit list (elst) offsets media time: the timeline was trimmed "
            "after encoding.")
    elif ev.has_edit_list:
        ev.findings.append(
            "Benign edit list present (B-frame reorder delay), not a trim.")

    if not ev.gop.gop_lengths:
        pass
    elif ev.gop.uniform:
        ev.findings.append(
            f"Uniform GOP of ~{ev.gop.gop_mean:.0f} frames: a single continuous "
            "encode (typical of a straight transcode).")
    else:
        ev.findings.append(
            f"Irregular GOP lengths (std {ev.gop.gop_std:.1f}): the stream was "
            "likely assembled from more than one encode (concat/splice).")

    if cut_times and fps > 0 and info.keyframe_indices:
        kf = set(info.keyframe_indices)
        cut_frames = _cut_frames(cut_times, fps)
        for cut_time, frame in zip(cut_times, cut_frames):
            near = any(abs(frame - k) <= 1 for k in kf)
            if near:
                ev.misaligned_cuts.append(cut_time)
            else:
                ev.spliced_cuts.append(cut_time)
        if ev.spliced_cuts:
            ev.findings.append(
                f"{len(ev.spliced_cuts)} cut(s) do not land on a keyframe: "
                "the material was likely re-encoded after cutting.")

    if ev.gop.uniform and ev.has_edit_list:
        ev.reencoded = True
    elif ev.gop.uniform:
        ev.reencoded = True
    elif ev.gop.gop_lengths and not ev.gop.uniform:
        ev.reencoded = False  # irregular GOPs are the camera/native pattern

    return ev
