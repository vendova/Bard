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

    from clip_diversity import ClipDiversity
    diversity = ClipDiversity(candidates)
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

            # Diversity: prefer unused moments, never repeat back-to-back.
            score += diversity.penalty(candidate, i)

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
        diversity.register(best_candidate, i)

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

    # Motion-on-beat: land the clip's peak-motion frame on the segment's
    # strongest accent so the visual movement reads as hitting the beat.
    beat_alignment = 0.0
    accent_offset = profile.get("anchor_offset")
    if accent_offset is not None and target in {"drop", "build", "rhythm"}:
        align = float(min(0.92, max(0.08, float(accent_offset) / max(final_duration, 1e-6))))
        beat_alignment = _clamp(profile.get("anchor_impact", 0.0), default=0.0)

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
        "anchor_time": profile.get("anchor_time"),
        "anchor_impact": profile.get("anchor_impact"),
        "beat_alignment": beat_alignment,
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


# A cut only earns a transition when it marks a real musical moment; ordinary
# cuts stay as hard cuts so transitions do not fire on every single boundary.
_TRANSITION_MIN_IMPORTANCE = 0.55

# Fraction of ordinary eligible cuts that are randomly downgraded to hard cuts,
# so the mix feels intentional instead of every highlight getting an effect.
# Strong landmarks (drop entries, section changes, hard hits) are never skipped.
_TRANSITION_SKIP_RATE = 0.2
_TRANSITION_LANDMARK_IMPORTANCE = 0.8

# Softening of the energy step when scoring importance (larger = more lenient).
_ENERGY_DELTA_SCALE = 0.5

# "Dip" transitions fade through a solid colour, which is visually heavy and
# gets tiring if it repeats. Cap how often they may appear in a rolling window.
_DIP_TRANSITIONS = {"fadeblack", "fadewhite"}
_DIP_MAX_PER_WINDOW = 2
_DIP_WINDOW_SECONDS = 20.0


def transition_importance(
    profile: Dict,
    next_profile: Optional[Dict],
    bar_anchor: bool = False,
    phrase_anchor: bool = False,
) -> float:
    """Score how much a cut *deserves* a transition, from 0 (hard cut) to 1.

    A transition adds meaning only at a musical landmark: a big energy step, a
    strong beat impact, a section change, a drop entry, or a bar/phrase anchor
    in the beat grid. Ordinary cuts that flow at a steady energy stay low and
    become hard cuts.
    """
    if next_profile is None:
        return 0.0

    seg_energy = compute_segment_energy(profile)
    next_energy = compute_segment_energy(next_profile)
    impact = _clamp(profile.get("impact", 0.5), default=0.5)
    section_type = profile.get("section_type", "body")
    next_section_type = next_profile.get("section_type", "body")
    target = profile.get("target", "flow")
    next_target = next_profile.get("target", "flow")

    # Smoothstep-ish ramp for the energy step (0 below 0.05, 1 above 0.45).
    energy_delta = abs(next_energy - seg_energy)
    delta_score = _clamp((energy_delta - 0.05) / _ENERGY_DELTA_SCALE, default=0.0)
    impact_score = _clamp((impact - 0.45) / 0.45, default=0.0)

    section_change = 0.0
    if next_section_type != section_type:
        # A cut that leaves the intro or reaches the outro is always a landmark.
        if section_type == "intro" or next_section_type == "outro":
            section_change = 0.85
        else:
            section_change = 0.7

    drop_entry = 0.0
    if next_target == "drop" and next_target != target and next_energy >= 0.6:
        drop_entry = 0.95

    importance = max(delta_score, drop_entry, section_change, 0.9 * impact_score)

    # Ordinary flow-to-flow cuts with no impact rarely need a transition.
    if target == "flow" and next_target == "flow" and impact < 0.55:
        importance *= 0.5

    # Beat-grid anchors: a phrase anchor (a musical "breath") is a natural place
    # for a transition; a bar anchor is a good one. This is a bonus on top of the
    # musical signals so anchored cuts win when otherwise similar.
    if phrase_anchor:
        importance += 0.16
    elif bar_anchor:
        importance += 0.08

    return _clamp(importance)


def _min_transition_gap(avg_energy: float) -> float:
    """Minimum spacing (seconds) between transitions for a given energy level.

    Intense passages may cut more often; calm passages space transitions out so
    they stay highlights. Cuts sit on beats, so this keeps transitions locked to
    the music instead of firing on every boundary.
    """
    return float(_clamp(6.0 - 6.5 * avg_energy, 1.5, 6.0))


def choose_transition_for_segment(
    profile: Dict,
    next_profile: Optional[Dict],
    index: int,
    prev_had_transition: bool = False,
    allow_dip: bool = True,
    bar_anchor: bool = False,
    phrase_anchor: bool = False,
    last_transition_time: float = -1e9,
) -> Optional[str]:
    """Return a transition type for the cut, or None to keep a hard cut.

    Combines :func:`transition_importance` (does this cut deserve a transition?)
    with the contextual transition pools (which transition fits?). ``cooldown``
    keeps ordinary cuts from chaining, ``allow_dip`` gates the heavy
    fade-through-black/white transitions, and the last two arguments enforce a
    beat-locked minimum gap between transitions.
    """
    if next_profile is None:
        return None

    importance = transition_importance(profile, next_profile, bar_anchor, phrase_anchor)
    if importance < _TRANSITION_MIN_IMPORTANCE:
        return None

    is_landmark = importance >= _TRANSITION_LANDMARK_IMPORTANCE

    # Cooldown: after a transition, only another landmark may follow directly,
    # so transitions read as highlights instead of running back to back.
    if prev_had_transition and not is_landmark:
        return None

    # Beat-locked spacing: respect the energy-dependent minimum gap unless the
    # cut is a genuine landmark (a drop or section change earns its moment).
    cut_time = float(profile.get("end", 0.0))
    avg_energy = (compute_segment_energy(profile)
                  + compute_segment_energy(next_profile)) * 0.5
    if not is_landmark and (cut_time - last_transition_time) < _min_transition_gap(avg_energy):
        return None

    rng = _stable_rng(index, "transition", round(importance, 3))

    # Occasionally drop an ordinary would-be transition so the effect stays a
    # highlight. Landmark cuts (drop entries, section changes, hard hits) are
    # always kept.
    if not is_landmark and rng.random() < _TRANSITION_SKIP_RATE:
        return None

    return _transition_type_for_context(profile, next_profile, rng, allow_dip)


def _transition_type_for_context(
    profile: Dict,
    next_profile: Dict,
    rng: random.Random,
    allow_dip: bool = True,
) -> str:
    """Pick the transition *type* that fits the musical context of a cut.

    When ``allow_dip`` is False, dip transitions (fade through black/white) are
    swapped for a non-dip alternative so they do not repeat too often.
    """
    seg_energy = compute_segment_energy(profile)
    next_energy = compute_segment_energy(next_profile)
    energy_delta = next_energy - seg_energy
    section_type = profile.get("section_type", "body")
    target = profile.get("target", "flow")

    # Section-specific overrides.
    if section_type == "intro":
        return _pick_from_pool(_TRANSITION_POOLS["intro"], rng, allow_dip)
    if section_type == "outro":
        return _pick_from_pool(_TRANSITION_POOLS["outro"], rng, allow_dip)

    # Big energy jump into a drop — dramatic transition.
    if next_profile.get("target") == "drop" and energy_delta > 0.15:
        return _pick_from_pool(_TRANSITION_POOLS["drop_hit"], rng, allow_dip)

    # Buildup section — smooth transitions.
    if section_type in {"bridge", "hook"} or target == "build":
        return _pick_from_pool(_TRANSITION_POOLS["build_up"], rng, allow_dip)

    # High energy both sides — energetic transitions.
    if seg_energy >= 0.65 and next_energy >= 0.65:
        return _pick_from_pool(_TRANSITION_POOLS["high_energy"], rng, allow_dip)

    # Big energy drop — calm after storm, use dissolve.
    if energy_delta < -0.20:
        return _pick_from_pool(_TRANSITION_POOLS["low_energy"], rng, allow_dip)

    # Low energy — soft transitions.
    if seg_energy <= 0.35 and next_energy <= 0.35:
        return _pick_from_pool(_TRANSITION_POOLS["low_energy"], rng, allow_dip)

    # Default — medium energy.
    return _pick_from_pool(_TRANSITION_POOLS["medium_energy"], rng, allow_dip)


def _pick_from_pool(pool: Sequence[str], rng: random.Random, allow_dip: bool) -> str:
    """Choose a transition from *pool*, honouring the dip-transition budget."""
    if allow_dip:
        return rng.choice(list(pool))
    non_dip = [t for t in pool if t not in _DIP_TRANSITIONS]
    if not non_dip:
        return rng.choice(list(pool))
    return rng.choice(non_dip)


def smart_transition_for_segment(
    profile: Dict,
    next_profile: Optional[Dict],
    index: int,
) -> str:
    """Pick a transition type for the cut (always returns a type).

    Kept for callers that want a definite transition; the adaptive planner uses
    :func:`choose_transition_for_segment`, which may return ``None`` for a hard
    cut.
    """
    if next_profile is None:
        return "fade"

    rng = _stable_rng(index, "transition-legacy",
                      round(compute_segment_energy(profile), 3),
                      round(compute_segment_energy(next_profile), 3))
    return _transition_type_for_context(profile, next_profile, rng)


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

    # Flash lands on the segment's strongest accent, not the clip start.
    anchor_offset = profile.get("anchor_offset")
    if anchor_offset is not None:
        overrides["flash_offset"] = float(anchor_offset)

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


def restrict_overrides_to_user_effects(
    overrides: Dict[str, Any], base_cfg: Any,
) -> Dict[str, Any]:
    """Drop intensity overrides for effects the user did not turn on.

    Smart Sync may compute intensity hints for every effect, but it must only
    *scale* the effects the user explicitly enabled. Otherwise Smart Sync would
    inject effects (vignette, color boost, flash, zoom punch, slow motion) that
    the user never asked for. The master toggle itself is still required, so
    this returns the overrides unchanged when the user enabled no effects.
    """
    if base_cfg is None:
        return {}
    if not getattr(base_cfg, "enabled", False):
        # Master toggle off: Smart Sync contributes no effect overrides.
        return {}
    if not (base_cfg.vignette or base_cfg.color_boost or base_cfg.flash_on_beat
            or base_cfg.zoom_punch or base_cfg.slow_motion):
        # Smart Sync enabled on its own (no effects selected): let it drive the
        # dynamic effects it is designed for. Everything is user-selected here.
        return dict(overrides)

    allowed: Dict[str, Any] = {}
    if base_cfg.vignette and "vignette_strength" in overrides:
        allowed["vignette_strength"] = overrides["vignette_strength"]
    if base_cfg.color_boost and "color_boost_amount" in overrides:
        allowed["color_boost_amount"] = overrides["color_boost_amount"]
    if base_cfg.flash_on_beat:
        for key in ("flash_on_beat", "flash_intensity", "flash_offset", "_flash_probability"):
            if key in overrides:
                allowed[key] = overrides[key]
    if base_cfg.zoom_punch:
        for key in ("zoom_punch_strength", "_zoom_punch_probability"):
            if key in overrides:
                allowed[key] = overrides[key]
    if base_cfg.slow_motion and "slow_motion_probability" in overrides:
        allowed["slow_motion_probability"] = overrides["slow_motion_probability"]
    return allowed


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

    if not base_cfg:
        return [], None
    # A cinematic look applies even when the effects master toggle is off, so
    # only short-circuit when there is neither a look nor enabled effects.
    if not base_cfg.enabled and not getattr(base_cfg, "look_name", ""):
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
    if "flash_offset" in overrides:
        cfg_kwargs["flash_offset"] = overrides["flash_offset"]

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

def snap_to_scene_boundary(start_time: float, scene_times: Sequence[float],
                           tolerance: float = 0.5) -> float:
    """Nudge a clip start toward the nearest shot boundary within *tolerance*.

    Avoids opening a clip mid-shot (which reads as a jarring jump cut). If no
    boundary is close enough, the original time is returned unchanged.
    """
    if not scene_times:
        return start_time
    nearest = min(scene_times, key=lambda t: abs(t - start_time))
    if abs(nearest - start_time) <= tolerance:
        return max(0.0, float(nearest))
    return start_time


def assign_camera_groups(video_files: Sequence[str],
                         offsets: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """Map each source to a camera-group id (multicam awareness).

    Files whose sync offset rounds to the same value are treated as the same
    physical camera / take, so the diversity pass can prefer switching between
    *different* cameras across adjacent cuts.
    """
    offsets = offsets or {}
    groups: Dict[str, str] = {}
    for path in video_files:
        off = offsets.get(path)
        key = f"cam@{round(off, 3):+.3f}" if off is not None else "cam@unknown"
        groups[path] = key
    return groups


def build_smart_sync_plan(
    profiles: List[Dict],
    candidates: List[Dict],
    video_files: Sequence[str],
    beat_info: Dict | None,
    scene_cuts_by_file: Optional[Dict[str, Sequence[float]]] = None,
    sync_offsets: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict], List[str], List[Dict[str, Any]]]:
    """Build the complete smart sync plan.

    Args:
        scene_cuts_by_file: optional shot boundaries per source file; when
            given, clip starts are snapped to shot boundaries so a clip does
            not open mid-shot.
        sync_offsets: optional per-file audio offsets used to tag clips with a
            camera group for multicam-aware diversity.

    Returns:
        planned_clips : List of planned-clip dicts (energy-matched).
        transitions   : List of transition types, one per cut (len = N-1).
        effect_overrides : List of per-clip effect override dicts.
    """
    # If no candidates from AI analysis, build basic ones.
    if not candidates and video_files:
        candidates = build_basic_candidates(video_files, beat_info)

    camera_groups = assign_camera_groups(video_files, sync_offsets)
    for candidate in candidates:
        candidate.setdefault("camera", camera_groups.get(candidate.get("video_file"), "cam@unknown"))

    planned_clips = smart_assign_clips(profiles, candidates)

    # Optional: snap each clip start to the nearest shot boundary.
    if scene_cuts_by_file:
        for clip in planned_clips:
            cuts = scene_cuts_by_file.get(clip.get("video_file"), [])
            clip["start_time"] = snap_to_scene_boundary(clip.get("start_time", 0.0), cuts)

    transitions = _place_transitions(profiles, beat_info)

    # Per-clip effect overrides.
    effect_overrides: List[Dict[str, Any]] = []
    for i, profile in enumerate(profiles):
        effect_overrides.append(smart_effect_overrides(profile, i))

    return planned_clips, transitions, effect_overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _place_transitions(profiles: List[Dict], beat_info: Dict | None) -> List[str]:
    """Choose where transitions go, one entry per cut ("" means hard cut).

    Cuts are scored by :func:`transition_importance` (energy jumps, drops,
    section changes, beat anchors). The most deserving cuts are placed first,
    then a beat-locked minimum gap — short in intense passages, wide in calm
    ones — and a rolling cap on heavy dip transitions keep the effect from
    spamming every boundary.
    """
    n = max(0, len(profiles) - 1)
    if n == 0:
        return []

    bar_flags, phrase_flags = _beat_anchor_flags(profiles, beat_info)

    cuts: List[Dict[str, Any]] = []
    for i in range(n):
        importance = transition_importance(
            profiles[i], profiles[i + 1], bar_flags[i], phrase_flags[i])
        if importance < _TRANSITION_MIN_IMPORTANCE:
            continue
        rng = _stable_rng(i, "transition", round(importance, 3))
        # Keep genuine landmarks (drops, section changes, hard hits); thin out
        # ordinary eligible cuts so the effect stays a highlight.
        if importance < _TRANSITION_LANDMARK_IMPORTANCE and rng.random() < _TRANSITION_SKIP_RATE:
            continue
        energy = (compute_segment_energy(profiles[i])
                  + compute_segment_energy(profiles[i + 1])) * 0.5
        cuts.append({
            "index": i,
            "time": float(profiles[i].get("end", 0.0)),
            "importance": importance,
            "energy": energy,
        })

    # Greedy placement, best cut first. Every clip contributes a baseline of
    # energy, so the average drives the minimum gap: busy passages transition
    # more often than calm ones.
    all_energy = [compute_segment_energy(p) for p in profiles] or [0.5]
    base_gap = _min_transition_gap(float(np.mean(all_energy)))

    transitions: List[str] = [""] * n
    placed: List[Dict[str, Any]] = []
    dips: List[float] = []
    for cut in sorted(cuts, key=lambda c: c["importance"], reverse=True):
        gap = max(1.0, base_gap * (0.6 if cut["importance"] >= _TRANSITION_LANDMARK_IMPORTANCE else 1.0))
        if any(abs(cut["time"] - p["time"]) < gap for p in placed):
            continue

        cut_time = cut["time"]
        allow_dip = sum(1 for t in dips if cut_time - t < _DIP_WINDOW_SECONDS) < _DIP_MAX_PER_WINDOW
        rng = _stable_rng(cut["index"], "type", round(cut["importance"], 3))
        t_type = _transition_type_for_context(
            profiles[cut["index"]], profiles[cut["index"] + 1], rng, allow_dip)

        transitions[cut["index"]] = t_type
        placed.append(cut)
        if t_type in _DIP_TRANSITIONS:
            dips.append(cut_time)

    return transitions


def _beat_anchor_flags(
    profiles: List[Dict], beat_info: Dict | None,
) -> Tuple[List[bool], List[bool]]:
    """Map each cut (the end of profile *i*) to the beat grid.

    Returns two boolean lists (one entry per cut): whether the cut lands on a
    bar anchor and on a phrase anchor. Transitions placed on these land with the
    music. Empty/False when beat data is unavailable.
    """
    n = max(0, len(profiles) - 1)
    bar_flags = [False] * n
    phrase_flags = [False] * n
    if not beat_info or n == 0:
        return bar_flags, phrase_flags

    times = np.asarray(beat_info.get("times", []), dtype=float)
    rhythm = beat_info.get("rhythm_data") or {}
    bar = np.asarray(rhythm.get("is_bar_anchor", []), dtype=bool)
    phrase = np.asarray(rhythm.get("is_phrase_anchor", []), dtype=bool)
    if times.size < 2 or bar.size != times.size or phrase.size != times.size:
        return bar_flags, phrase_flags

    half_beat = float(np.median(np.diff(times))) * 0.5
    for i in range(n):
        cut_time = float(profiles[i].get("end", 0.0))
        j = int(np.searchsorted(times, cut_time))
        candidates = [k for k in (j - 1, j) if 0 <= k < times.size]
        nearest = min(candidates, key=lambda k: abs(times[k] - cut_time))
        if abs(times[nearest] - cut_time) <= half_beat:
            bar_flags[i] = bool(bar[nearest])
            phrase_flags[i] = bool(phrase[nearest])
    return bar_flags, phrase_flags


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
