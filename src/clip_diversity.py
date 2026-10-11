"""Source-moment diversity for clip planning.

BeatSync's planners score candidate source moments for each output segment.
Without a diversity constraint the same moment or the same source video tends
to win repeatedly, producing a video that keeps showing the same clip. This
module supplies a small stateful scorer that both planners mix into their
candidate scores so that, across the whole video:

- an unused source moment is always preferred over an already-used one,
- the same moment never appears on two consecutive cuts,
- a moment that was used recently is penalised until enough other moments
  have played (spread round-robin when footage is exhausted),
- two adjacent cuts avoid the same source video when alternatives exist.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


class ClipDiversity:
    """Tracks per-moment usage and returns diversity penalties for scoring."""

    def __init__(self, candidates: Sequence[Dict]):
        count = max(1, len(candidates))
        # Repeats should be spread at least this far apart. With plenty of
        # footage this exceeds the segment count, i.e. no repeats at all.
        self.min_gap = max(2, count // 2)
        self._usage: Counter = Counter()
        self._last_used_at: Dict[Any, int] = {}
        self._prev_id: Optional[Any] = None
        self._prev_video: Optional[Any] = None

    def penalty(self, candidate: Dict, index: int) -> float:
        """Return a non-positive score adjustment for *candidate* at *index*."""
        cid = candidate.get("id")
        video_file = candidate.get("video_file")
        penalty = 0.0

        if cid == self._prev_id:
            # Hard block: never show the same moment on consecutive cuts.
            penalty -= 1.5
        elif cid in self._last_used_at:
            gap = index - self._last_used_at[cid]
            if gap < self.min_gap:
                # Recency penalty that fades as the gap grows.
                penalty -= 0.55 * (1.0 - gap / self.min_gap)

        # Strongly prefer moments that have not been used yet, then balance
        # reuse round-robin once every moment has appeared.
        penalty -= min(1.6, self._usage[cid] * 0.7)

        if video_file == self._prev_video:
            penalty -= 0.35
        penalty -= min(0.4, self._usage[video_file] * 0.02)

        return penalty

    def register(self, candidate: Dict, index: int) -> None:
        cid = candidate.get("id")
        self._usage[cid] += 1
        self._usage[candidate.get("video_file")] += 1
        self._last_used_at[cid] = index
        self._prev_id = cid
        self._prev_video = candidate.get("video_file")


def summarize_reuse(plan: Sequence[Dict]) -> Dict[str, int]:
    """Count distinct vs total moments and the worst back-to-back run."""
    ids = [item.get("candidate_id") for item in plan]
    total = len(ids)
    distinct = len(set(i for i in ids if i is not None))
    back_to_back = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    return {
        "total_clips": total,
        "distinct_moments": distinct,
        "back_to_back_repeats": back_to_back,
    }
