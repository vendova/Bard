"""
Smart Sync Engine — AI-driven clip-to-beat energy matching with adaptive
transitions and per-clip effect intensities.

This module augments the existing visual planner with three capabilities:

1. **Clip energy profiling** — maps raw video metrics (motion, contrast,
   saturation, etc.) to a unified 0-1 energy score, even when Qwen AI
   semantic analysis is unavailable.

2. **Energy-matched assignment** — shuffles and assigns source clips to
   beat segments so that high-energy beats get high-energy clips and
   low-energy beats get calm clips, while maintaining diversity (no
   repeated clip back-to-back, balanced source usage).

3. **Adaptive transitions & effects** — picks a transition type per cut
   based on the energy delta between adjacent segments, and scales
   per-clip effect intensities (flash, vignette, color boost, etc.)
   according to the segment's beat energy.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

def compute_clip_energy(candidate: Dict) -> float:
    """Map a video candidate's metrics to a 0-1 energy score.

    Uses action, motion, tension, contrast and saturation when available.
    Falls back gracefully when only partial metrics exist.
    """
    action = _clamp(candidate.get("action_score", 0.0))
    motion = _clamp(candidate.get("motion", 0.0))
    tension = _clamp(candidate.get("tension_score", 0.0))
    contrast = _clamp(candidate.get("contrast", 0.0), default=0.5)
    saturation = _clamp(candidate.get("saturation", 0.0), default=0.5)
    semantic = candidate.get("semantic") or {}
    combat = _clamp(semantic.get("combat", 0.0))
    chase = _clamp(semantic.get("chase", 0.0))
    explosion = _clamp(semantic.get("explosion", 0.0))

    energy = (
        0.30 * action
        + 0.20 * motion
        + 0.12 * tension
        + 0.08 * contrast
        + 0.06 * saturation
        + 0.10 * combat
        + 0.08 * chase
        + 0.06 * explosion
    )
    return _clamp(energy)


def compute_segment_energy(profile: Dict) -> float:
    """Map a segment's audio profile to a 0-1 energy score.

    Combines wave (spectral energy), impact (transient strength), and
    rhythm (combined rhythmic intensity).
    """
    wave = _clamp(profile.get("wave", 0.5), default=0.5)
    impact = _clamp(profile.get("impact", 0.5), default=0.5)
    rhythm = _clamp(profile.get("rhythm", 0.5), default=0.5)
    novelty = _clamp(profile.get("novelty", 0.4), default=0.4)
    arc = _clamp(profile.get("arc", 0.5), default=0.5)

    return _clamp(0.36 * wave + 0.28 * impact + 0.20 * rhythm + 0.10 * novelty + 0.06 * arc)


# ---------------------------------------------------------------------------
# Smart clip assignment
# ---------------------------------------------------------------------------

def smart_assign_clips(
    profiles: List[Dict],
    candidates: List[Dict],
) -> List[Dict]:
    """Assign clips to segments using energy matching + diversity constraints.

    Each profile receives the candidate whose energy best matches the
    segment's energy, with penalties for recent reuse and overuse.

    Returns a list of planned-clip dicts (same shape as stage6's
    _materialize_clip output) with an added ``clip_energy`` field.
    """
    if not profiles or not candidates:
        return []

    # Pre-compute clip energies.
    clip_energies = {c.get("id"): compute_clip_energy(c) for c in candidates}
    seg_energies = [compute_segment_energy(p) for p in profiles]

    recent_ids: deque = deque(maxlen=8)
    recent_videos: deque = deque(maxlen=4)
    usage: Counter = Counter()
    planned: List[Dict] = []

    for i, profile in enumerate(profiles):
        seg_energy = seg_energies[i]
        best_candidate = None
        best_score = -999.0
        rng = _stable_rng(i, profile.get("target", "flow"), round(seg_energy, 3))

        for candidate in candidates:
            cid = candidate.get("id")
            clip_energy = clip_energies.get(cid, 0.5)

            # Energy match: closer energies = higher score.
            energy_diff = abs(seg_energy - clip_energy)
            energy_match = 1.0 - energy_diff  # 0..1

            # Base score from the existing planner's semantic scoring.
            base = _semantic_base_score(candidate, profile)

            score = (
                0.45 * energy_match
                + 0.35 * base
                + 0.20 * _clamp(candidate.get("quality_score", 0.5), default=0.5)
            )

            # Diversity penalties.
            if cid in recent_ids:
                score -= 0.30
            video_file = candidate.get("video_file")
            if video_file in recent_videos:
                score -= 0.12
            score -= min(0.30, usage[cid] * 0.12)
            score -= min(0.20, usage[video_file] * 0.015)

            # Duration feasibility.
            required = max(0.05, profile.get("duration", 1.0))
            cand_dur = max(0.05, float(candidate.get("duration", required)))
            if cand_dur < required * 0.55:
                score -= 0.18

            # Small jitter for variety.
            score += rng.random() * 0.02

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if not best_candidate:
            continue

        planned_clip = _materialize_smart_clip(best_candidate, profile, i, seg_energy)
        planned.append(planned_clip)
        recent_ids.append(best_candidate.get("id"))
        recent_videos.append(best_candidate.get("video_file"))
        usage[best_candidate.get("id")] += 1
        usage[best_candidate.get("video_file")] += 1

    return planned


def _semantic_base_score(candidate: Dict, profile: Dict) -> float:
    """Lightweight version of stage6's _score_candidate for energy context."""
    target = profile.get("target", "flow")
    semantic = candidate.get("semantic") or {}
    action = _clamp(candidate.get("action_score", semantic.get("action_intensity", 0.0)))
    beauty = _clamp(candidate.get("beauty_score", semantic.get("beauty_score", 0.0)))
    tension = _clamp(candidate.get("tension_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    motion = _clamp(candidate.get("motion", semantic.get("camera_motion", 0.0)))
    quality = _clamp(candidate.get("quality_score", 0.5), default=0.5)

    if target == "drop":
        return _clamp(0.46 * action + 0.16 * motion + 0.12 * quality)
    if target == "soft":
        return _clamp(0.45 * beauty + 0.18 * soft + 0.14 * (1.0 - action) + 0.10 * quality)
    if target == "build":
        return _clamp(0.40 * tension + 0.18 * motion + 0.15 * action + 0.13 * quality)
    if target == "rhythm":
        return _clamp(0.30 * action + 0.24 * motion + 0.18 * quality + 0.16 * tension)
    return _clamp(0.28 * quality + 0.24 * beauty + 0.20 * action + 0.16 * tension + 0.12 * soft)


def _materialize_smart_clip(candidate: Dict, profile: Dict, index: int, seg_energy: float) -> Dict:
    """Build the planned-clip dict, mirroring stage6's _materialize_clip."""
    final_duration = max(0.05, float(profile.get("duration", 1.0)))
    source_duration = final_duration
    video_duration = max(source_duration, float(candidate.get("video_duration", source_duration)))
    target = profile.get("target", "flow")

    if target == "drop":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.36
    elif target == "soft":
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.50
    elif target == "build":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.48
    else:
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.44

    start_time = anchor - source_duration * align
    start_time = max(0.0, min(start_time, max(0.0, video_duration - source_duration)))

    return {
        "index": index,
        "video_file": candidate.get("video_file"),
        "source_name": candidate.get("source_name"),
        "start_time": start_time,
        "source_duration": source_duration,
        "final_duration": final_duration,
        "target": target,
        "candidate_id": candidate.get("id"),
        "tags": list(candidate.get("tags", [])),
        "ai_analyzed": bool(candidate.get("ai_analyzed")),
        "audio_start": profile.get("start"),
        "audio_end": profile.get("end"),
        "wave": profile.get("wave"),
        "impact": profile.get("impact"),
        "segment_energy": seg_energy,
        "clip_energy": compute_clip_energy(candidate),
    }


# ---------------------------------------------------------------------------
# Fallback: build candidates from raw video files (no AI analysis)
# ---------------------------------------------------------------------------

def build_basic_candidates(video_files: Sequence[str], beat_info: Dict | None) -> List[Dict]:
    """Construct lightweight candidates when full video analysis is unavailable.

    Probes each video for duration/fps/resolution and assigns a neutral
    energy of 0.5 so the energy matcher still produces a smart shuffle
    (diversity-based) rather than random.choice.
    """
    from ffmpeg_processing import get_video_duration, get_video_fps, get_video_resolution

    beat_info = beat_info or {}
    candidates: List[Dict] = []
    for video_file in video_files:
        try:
            duration = max(1.0, float(get_video_duration(video_file)))
            fps = max(1.0, float(get_video_fps(video_file)))
            width, height = get_video_resolution(video_file)
        except Exception:
            duration, fps, width, height = 10.0, 30.0, 1920, 1080

        # Create multiple sampling windows per video.
        num_windows = max(3, min(20, int(duration / 3.0)))
        for w_idx in range(num_windows):
            seg_start = (duration / num_windows) * w_idx
            seg_end = min(duration, seg_start + duration / num_windows)
            seg_dur = max(0.5, seg_end - seg_start)
            cid = f"basic_{_hash_text(os.path.abspath(video_file), 8)}_{w_idx:04d}"
            candidates.append({
                "id": cid,
                "video_file": os.path.abspath(video_file),
                "source_name": os.path.basename(video_file),
                "video_duration": duration,
                "start": seg_start,
                "end": seg_end,
                "duration": seg_dur,
                "center": seg_start + seg_dur * 0.5,
                "peak_time": seg_start + seg_dur * 0.5,
                "motion": 0.5,
                "brightness": 0.5,
                "contrast": 0.5,
                "saturation": 0.5,
                "sharpness": 0.5,
                "colorfulness": 0.5,
                "quality_score": 0.5,
                "action_score": 0.5,
                "beauty_score": 0.5,
                "tension_score": 0.4,
                "soft_score": 0.4,
                "editorial_score": 0.5,
                "tags": ["flow"],
                "semantic": {},
                "ai_analyzed": False,
            })
    return candidates


# ---------------------------------------------------------------------------
# Adaptive transitions
# ---------------------------------------------------------------------------

# Transition pools by energy context.
_TRANSITION_POOLS: Dict[str, List[str]] = {
    "high_energy": ["slideleft", "slideright", "zoomin", "wipeleft", "wiperight", "circleopen"],
    "medium_energy": ["fade", "dissolve", "smoothleft", "smoothright", "radial"],
    "low_energy": ["fade", "fadeblack", "fadewhite", "dissolve"],
    "intro": ["fadeblack", "fade"],
    "outro": ["fadeblack", "fadewhite", "fade"],
    "drop_hit": ["zoomin", "slideleft", "slideright", "circleopen"],
    "build_up": ["smoothup", "smoothleft", "smoothright", "fade"],
}


def smart_transition_for_segment(
    profile: Dict,
    next_profile: Optional[Dict],
    index: int,
) -> str:
    """Pick a transition type for the cut between *profile* and *next_profile*.

    The choice depends on:
    - Section type (intro/outro/drop/build)
    - Energy delta between segments
    - Beat impact strength
    """
    if next_profile is None:
        return "fade"

    seg_energy = compute_segment_energy(profile)
    next_energy = compute_segment_energy(next_profile)
    energy_delta = next_energy - seg_energy
    impact = _clamp(profile.get("impact", 0.5), default=0.5)
    section_type = profile.get("section_type", "body")

    rng = _stable_rng(index, "transition", round(seg_energy, 3), round(next_energy, 3))

    # Section-specific overrides.
    if section_type == "intro":
        return rng.choice(_TRANSITION_POOLS["intro"])
    if section_type == "outro":
        return rng.choice(_TRANSITION_POOLS["outro"])

    # Big energy jump into a drop — dramatic transition.
    if next_profile.get("target") == "drop" and energy_delta > 0.15:
        return rng.choice(_TRANSITION_POOLS["drop_hit"])

    # Buildup section — smooth transitions.
    if section_type in {"bridge", "hook"} or profile.get("target") == "build":
        return rng.choice(_TRANSITION_POOLS["build_up"])

    # High energy both sides — energetic transitions.
    if seg_energy >= 0.65 and next_energy >= 0.65:
        return rng.choice(_TRANSITION_POOLS["high_energy"])

    # Big energy drop — calm after storm, use dissolve.
    if energy_delta < -0.20:
        return rng.choice(_TRANSITION_POOLS["low_energy"])

    # Low energy — soft transitions.
    if seg_energy <= 0.35 and next_energy <= 0.35:
        return rng.choice(_TRANSITION_POOLS["low_energy"])

    # Default — medium energy.
    return rng.choice(_TRANSITION_POOLS["medium_energy"])


def smart_transition_duration(seg_energy: float, next_energy: float, base_duration: float) -> float:
    """Adjust transition duration based on energy.

    High-energy cuts get shorter, snappier transitions; calm sections
    get longer, more gradual transitions.
    """
    avg_energy = (seg_energy + next_energy) * 0.5
    if avg_energy >= 0.7:
        return max(0.15, base_duration * 0.6)
    if avg_energy <= 0.3:
        return min(0.8, base_duration * 1.4)
    return base_duration


# ---------------------------------------------------------------------------
# Adaptive per-clip effect intensities
# ---------------------------------------------------------------------------

def smart_effect_overrides(profile: Dict, clip_index: int) -> Dict[str, Any]:
    """Return per-clip effect overrides based on segment energy.

    The returned dict can be merged into an EffectsConfig clone to scale
    effect intensity dynamically:
    - flash_intensity: stronger on high-impact beats
    - vignette_strength: stronger on soft/emotional segments
    - color_boost_amount: stronger on high-energy drops
    - zoom_punch_strength: stronger on strong impacts
    - slow_motion_probability: trigger on impactful moments
    """
    seg_energy = compute_segment_energy(profile)
    impact = _clamp(profile.get("impact", 0.5), default=0.5)
    target = profile.get("target", "flow")
    section_type = profile.get("section_type", "body")

    rng = _stable_rng(clip_index, "effects", round(seg_energy, 3))

    overrides: Dict[str, Any] = {}

    # Flash: strong on drops and high-impact beats.
    flash_prob = 0.0
    flash_int = 0.0
    if target == "drop" or impact >= 0.7:
        flash_prob = 0.6
        flash_int = min(1.0, 0.4 + impact * 0.5)
    elif impact >= 0.5:
        flash_prob = 0.25
        flash_int = 0.3 + impact * 0.3
    overrides["flash_on_beat"] = flash_prob > 0.0
    overrides["flash_intensity"] = flash_int
    overrides["_flash_probability"] = flash_prob  # used by smart_build_filters

    # Vignette: stronger on soft/emotional segments.
    if target == "soft" or section_type in {"breakdown", "outro"}:
        overrides["vignette_strength"] = min(0.8, 0.4 + (1.0 - seg_energy) * 0.4)
    elif target == "drop":
        overrides["vignette_strength"] = 0.3
    else:
        overrides["vignette_strength"] = 0.35

    # Color boost: stronger on high-energy drops.
    if target == "drop" and seg_energy >= 0.6:
        overrides["color_boost_amount"] = min(1.8, 1.2 + seg_energy * 0.5)
    elif target == "soft":
        overrides["color_boost_amount"] = 1.05
    else:
        overrides["color_boost_amount"] = 1.15

    # Zoom punch: sync to strong impacts.
    if impact >= 0.65:
        overrides["zoom_punch_strength"] = min(1.15, 1.04 + impact * 0.08)
        overrides["_zoom_punch_probability"] = 0.7
    else:
        overrides["zoom_punch_strength"] = 1.06
        overrides["_zoom_punch_probability"] = 0.3

    # Slow motion: trigger on high-impact, non-drop moments (dramatic effect).
    if impact >= 0.75 and target != "drop":
        overrides["slow_motion_probability"] = 0.3
    elif target == "soft":
        overrides["slow_motion_probability"] = 0.2
    else:
        overrides["slow_motion_probability"] = 0.08

    # Glitch: rare, only on very high energy.
    if seg_energy >= 0.8 and rng.random() < 0.3:
        overrides["_glitch_probability"] = 0.4
    else:
        overrides["_glitch_probability"] = 0.0

    return overrides


def smart_build_clip_effect_filters(
    base_cfg: Any,
    overrides: Dict[str, Any],
    clip_index: int,
    clip_duration: float,
    fps: float,
    target_size: Tuple[int, int],
) -> Tuple[List[str], Optional[float]]:
    """Build per-clip effect filters with smart intensity overrides.

    Wraps the existing ``build_clip_effect_filters`` but applies per-clip
    overrides from ``smart_effect_overrides`` before generating filters.
    """
    from dataclasses import replace
    from video_effects import build_clip_effect_filters, EffectsConfig

    if not base_cfg or not base_cfg.enabled:
        return [], None

    # Clone the config and apply overrides.
    cfg_kwargs: Dict[str, Any] = {}
    if "vignette_strength" in overrides:
        cfg_kwargs["vignette_strength"] = overrides["vignette_strength"]
    if "color_boost_amount" in overrides:
        cfg_kwargs["color_boost_amount"] = overrides["color_boost_amount"]
    if "zoom_punch_strength" in overrides:
        cfg_kwargs["zoom_punch_strength"] = overrides["zoom_punch_strength"]
    if "slow_motion_probability" in overrides:
        cfg_kwargs["slow_motion_probability"] = overrides["slow_motion_probability"]
    if "flash_intensity" in overrides:
        cfg_kwargs["flash_intensity"] = overrides["flash_intensity"]

    if cfg_kwargs:
        effective_cfg = replace(base_cfg, **cfg_kwargs)
    else:
        effective_cfg = base_cfg

    filters, speed_factor = build_clip_effect_filters(
        effective_cfg, clip_index, clip_duration, fps, target_size,
    )

    # Inject smart flash probability (only flash on some clips, not all).
    flash_prob = overrides.get("_flash_probability", 0.0)
    if flash_prob > 0.0:
        rng = _stable_rng(clip_index, "flash", round(flash_prob, 3))
        if rng.random() > flash_prob:
            # Remove the flash drawbox filter if present.
            filters = [f for f in filters if "drawbox" not in f]

    return filters, speed_factor


# ---------------------------------------------------------------------------
# Full smart sync plan
# ---------------------------------------------------------------------------

def build_smart_sync_plan(
    profiles: List[Dict],
    candidates: List[Dict],
    video_files: Sequence[str],
    beat_info: Dict | None,
) -> Tuple[List[Dict], List[str], List[Dict[str, Any]]]:
    """Build the complete smart sync plan.

    Returns:
        planned_clips : List of planned-clip dicts (energy-matched).
        transitions   : List of transition types, one per cut (len = N-1).
        effect_overrides : List of per-clip effect override dicts.
    """
    # If no candidates from AI analysis, build basic ones.
    if not candidates and video_files:
        candidates = build_basic_candidates(video_files, beat_info)

    planned_clips = smart_assign_clips(profiles, candidates)

    # Per-cut transitions.
    transitions: List[str] = []
    for i in range(len(profiles) - 1):
        t_type = smart_transition_for_segment(profiles[i], profiles[i + 1], i)
        transitions.append(t_type)

    # Per-clip effect overrides.
    effect_overrides: List[Dict[str, Any]] = []
    for i, profile in enumerate(profiles):
        effect_overrides.append(smart_effect_overrides(profile, i))

    return planned_clips, transitions, effect_overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if not np.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _stable_rng(*parts) -> random.Random:
    raw = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    return random.Random(seed)


def _hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]
