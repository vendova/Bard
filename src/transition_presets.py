"""
100 Smart Transition Presets for music video cuts.

Each preset defines a named combination of transition strategies that the
smart sync engine uses to pick per-cut transitions. The presets vary the
transition pool, duration scaling, and selection logic to produce visually
distinct rhythm profiles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# All available xfade transition types
ALL_TRANSITIONS = [
    "fade", "fadeblack", "fadewhite",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circleopen", "circleclose",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "dissolve", "radial", "zoomin",
]


@dataclass
class TransitionPreset:
    name: str
    high_energy_pool: List[str] = field(default_factory=lambda: ["slideleft", "zoomin", "circleopen"])
    medium_energy_pool: List[str] = field(default_factory=lambda: ["fade", "dissolve", "smoothleft"])
    low_energy_pool: List[str] = field(default_factory=lambda: ["fade", "fadeblack", "dissolve"])
    intro_pool: List[str] = field(default_factory=lambda: ["fadeblack", "fade"])
    outro_pool: List[str] = field(default_factory=lambda: ["fadeblack", "fadewhite"])
    drop_pool: List[str] = field(default_factory=lambda: ["zoomin", "slideleft", "slideright"])
    build_pool: List[str] = field(default_factory=lambda: ["smoothup", "smoothleft"])
    base_duration: float = 0.35
    high_energy_dur_mult: float = 0.6    # shorter transitions for high energy
    low_energy_dur_mult: float = 1.4     # longer transitions for calm
    variety_bias: float = 0.5            # 0 = uniform, 1 = maximum variety

    def pick(self, context: str, seg_energy: float, next_energy: float,
             index: int, rng: random.Random) -> str:
        """Pick a transition for a given context."""
        pools = {
            "intro": self.intro_pool,
            "outro": self.outro_pool,
            "drop": self.drop_pool,
            "build": self.build_pool,
            "high": self.high_energy_pool,
            "medium": self.medium_energy_pool,
            "low": self.low_energy_pool,
        }
        pool = pools.get(context, self.medium_energy_pool)
        if not pool:
            pool = ["fade"]

        # With variety_bias, avoid repeating the last transition.
        if self.variety_bias > 0 and len(pool) > 1:
            # Deterministic-ish pick based on index + rng
            pick_idx = int(rng.random() * len(pool))
            return pool[pick_idx]
        return pool[0]

    def duration_for(self, seg_energy: float, next_energy: float) -> float:
        avg = (seg_energy + next_energy) * 0.5
        if avg >= 0.65:
            return max(0.12, self.base_duration * self.high_energy_dur_mult)
        if avg <= 0.3:
            return min(0.8, self.base_duration * self.low_energy_dur_mult)
        return self.base_duration


def _p(name, hi, med, lo, intro=None, outro=None, drop=None, build=None,
       dur=0.35, hi_mult=0.6, lo_mult=1.4, variety=0.5):
    return TransitionPreset(
        name=name,
        high_energy_pool=hi, medium_energy_pool=med, low_energy_pool=lo,
        intro_pool=intro or ["fadeblack", "fade"],
        outro_pool=outro or ["fadeblack", "fadewhite"],
        drop_pool=drop or ["zoomin", "slideleft"],
        build_pool=build or ["smoothup", "smoothleft"],
        base_duration=dur, high_energy_dur_mult=hi_mult,
        low_energy_dur_mult=lo_mult, variety_bias=variety,
    )


# ---------------------------------------------------------------------------
# 100 Transition Presets
# ---------------------------------------------------------------------------

TRANSITION_PRESETS: List[TransitionPreset] = [
    # 1-10: Balanced presets
    _p("Balanced Flow", ["slideleft", "zoomin", "circleopen"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "dissolve"]),
    _p("Cinematic Blend", ["zoomin", "radial", "dissolve"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite"], dur=0.4),
    _p("Energy Wave", ["slideleft", "slideright", "zoomin"], ["smoothleft", "smoothright", "fade"], ["fade", "dissolve"], hi_mult=0.5),
    _p("Smooth Operator", ["smoothleft", "smoothright", "smoothup"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack"], hi_mult=0.8, lo_mult=1.5),
    _p("Hard Cut Pro", ["slideleft", "slideright", "wipeleft"], ["slideleft", "fade", "dissolve"], ["fade", "dissolve"], hi_mult=0.3, dur=0.25),
    _p("Gentle Touch", ["fade", "dissolve", "smoothleft"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite"], hi_mult=0.9, lo_mult=1.6, dur=0.45),
    _p("Dynamic Mix", ["zoomin", "circleopen", "slideleft"], ["radial", "dissolve", "fade"], ["fade", "fadewhite", "dissolve"], variety=0.8),
    _p("Rhythm Rider", ["wipeleft", "wiperight", "zoomin"], ["slideleft", "fade", "smoothleft"], ["fade", "dissolve"], hi_mult=0.4),
    _p("Flow Master", ["smoothleft", "smoothright", "fade"], ["fade", "dissolve", "smoothup"], ["fade", "fadeblack", "dissolve"]),
    _p("Pulse Beat", ["zoomin", "slideleft", "circleopen"], ["dissolve", "radial", "fade"], ["fade", "fadewhite"], hi_mult=0.5, variety=0.7),

    # 11-20: High-energy presets
    _p("Adrenaline Rush", ["slideleft", "slideright", "zoomin", "wipeleft"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.3, dur=0.2),
    _p("Max Energy", ["zoomin", "slideleft", "circleopen", "wiperight"], ["slideleft", "radial", "fade"], ["fade", "dissolve"], hi_mult=0.25),
    _p("Electric Storm", ["wipeleft", "wiperight", "zoomin", "slideup"], ["slideleft", "slideright", "dissolve"], ["fade", "dissolve"], hi_mult=0.3, variety=0.9),
    _p("Hyper Drive", ["zoomin", "circleopen", "slideleft", "slideright"], ["radial", "zoomin", "fade"], ["fade", "fadewhite"], hi_mult=0.2, dur=0.18),
    _p("Turbo Cut", ["slideleft", "slideright", "wipeleft", "wiperight"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.2, dur=0.15),
    _p("Power Surge", ["zoomin", "circleopen", "radial", "slideleft"], ["zoomin", "radial", "dissolve"], ["fade", "dissolve", "fadewhite"], hi_mult=0.3),
    _p("Bass Drop Special", ["zoomin", "slideleft", "circleopen", "wipeleft"], ["slideleft", "slideright", "dissolve"], ["fade", "fadeblack", "dissolve"], drop=["zoomin", "slideleft", "circleopen", "wipeleft", "wiperight"], hi_mult=0.25),
    _p("Beat Smasher", ["wipeleft", "wiperight", "slideup", "slidedown"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.2, variety=0.9),
    _p("Rage Mode", ["zoomin", "slideleft", "slideright", "wipeleft", "wiperight"], ["slideleft", "slideright", "zoomin", "fade"], ["fade", "dissolve"], hi_mult=0.15, dur=0.12),
    _p("Fury Flow", ["circleopen", "circleclose", "zoomin", "slideleft"], ["radial", "zoomin", "dissolve", "fade"], ["fade", "fadewhite", "dissolve"], hi_mult=0.25, variety=0.8),

    # 21-30: Smooth/cinematic presets
    _p("Silk Screen", ["smoothleft", "smoothright", "smoothup", "smoothdown"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.7, lo_mult=1.5, dur=0.5),
    _p("Cinema Dream", ["dissolve", "fade", "radial"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.8, lo_mult=1.6, dur=0.55),
    _p("Velvet Touch", ["smoothleft", "smoothright", "fade", "dissolve"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "dissolve"], hi_mult=0.85, dur=0.5),
    _p("Ocean Calm", ["fade", "dissolve", "smoothleft", "smoothright"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.9, lo_mult=1.7, dur=0.6),
    _p("Whisper Flow", ["fade", "dissolve", "smoothup"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=1.0, lo_mult=1.8, dur=0.65),
    _p("Classic Cinema", ["dissolve", "fade", "fadeblack"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.8, dur=0.5),
    _p("Art House", ["dissolve", "fade", "radial", "smoothleft"], ["fade", "dissolve"], ["fade", "fadeblack", "dissolve"], hi_mult=0.85, variety=0.6, dur=0.5),
    _p("Dream Sequence", ["fade", "fadewhite", "dissolve", "smoothleft"], ["fade", "dissolve", "fadewhite"], ["fade", "fadewhite", "dissolve"], hi_mult=0.9, lo_mult=1.6, dur=0.55),
    _p("Slow Burn", ["dissolve", "fade", "smoothleft", "smoothright"], ["fade", "dissolve"], ["fade", "fadeblack", "dissolve"], hi_mult=0.75, lo_mult=1.5, dur=0.5),
    _p("Gentle Wave", ["smoothleft", "smoothright", "smoothup", "smoothdown", "fade"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.85, lo_mult=1.5, dur=0.5),

    # 31-40: Circle/wipe focused
    _p("Circle of Life", ["circleopen", "circleclose", "zoomin"], ["circleopen", "fade", "dissolve"], ["fade", "dissolve", "fadewhite"]),
    _p("Wipe Out", ["wipeleft", "wiperight", "wipeup", "wipedown"], ["wipeleft", "wiperight", "fade"], ["fade", "dissolve"], hi_mult=0.5),
    _p("Circle Master", ["circleopen", "circleclose", "radial", "zoomin"], ["circleopen", "radial", "fade"], ["fade", "dissolve", "fadewhite"], variety=0.7),
    _p("Wipe Wizard", ["wipeleft", "wiperight", "wipeup", "wipedown", "slideleft"], ["wipeleft", "wiperight", "fade", "dissolve"], ["fade", "dissolve"], hi_mult=0.4, variety=0.8),
    _p("Radial Sun", ["radial", "circleopen", "zoomin", "dissolve"], ["radial", "fade", "dissolve"], ["fade", "fadewhite", "dissolve"]),
    _p("Circle Dance", ["circleopen", "circleclose", "fade", "dissolve"], ["circleopen", "fade", "dissolve"], ["fade", "fadeblack", "dissolve"], hi_mult=0.7),
    _p("Wipe Surge", ["wipeleft", "wiperight", "zoomin", "slideleft"], ["wipeleft", "wiperight", "fade"], ["fade", "dissolve"], hi_mult=0.3, variety=0.8),
    _p("Circle Pulse", ["circleopen", "circleclose", "zoomin", "radial", "slideleft"], ["circleopen", "radial", "fade"], ["fade", "fadewhite", "dissolve"], hi_mult=0.5, variety=0.7),
    _p("Wipe and Fade", ["wipeleft", "wiperight", "fade", "dissolve"], ["wipeleft", "fade", "dissolve"], ["fade", "fadeblack", "dissolve"]),
    _p("Radial Wave", ["radial", "circleopen", "circleclose", "dissolve"], ["radial", "fade", "dissolve"], ["fade", "fadewhite", "dissolve"], hi_mult=0.6),

    # 41-50: Slide focused
    _p("Slide Master", ["slideleft", "slideright", "slideup", "slidedown"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.4),
    _p("Slide and Glide", ["slideleft", "slideright", "smoothleft", "smoothright"], ["slideleft", "fade", "smoothleft"], ["fade", "dissolve", "fadeblack"]),
    _p("Directional Flow", ["slideleft", "slideright", "slideup", "slidedown", "zoomin"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.35, variety=0.8),
    _p("Slide Surge", ["slideleft", "slideright", "wipeleft", "zoomin"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.3),
    _p("Smooth Slide", ["smoothleft", "smoothright", "smoothup", "smoothdown", "slideleft"], ["smoothleft", "fade", "dissolve"], ["fade", "fadeblack", "dissolve"], hi_mult=0.6),
    _p("Slide Energy", ["slideleft", "slideright", "zoomin", "circleopen"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.35, variety=0.7),
    _p("Horizontal Hustle", ["slideleft", "slideright", "wipeleft", "wiperight"], ["slideleft", "slideright", "fade"], ["fade", "dissolve"], hi_mult=0.3),
    _p("Vertical Vibe", ["slideup", "slidedown", "zoomin", "circleopen"], ["slideup", "slidedown", "fade"], ["fade", "dissolve"], hi_mult=0.4),
    _p("Slide and Zoom", ["slideleft", "slideright", "zoomin", "circleopen"], ["slideleft", "fade", "zoomin"], ["fade", "dissolve", "fadewhite"], hi_mult=0.35),
    _p("Smooth Operator X", ["smoothleft", "smoothright", "smoothup", "smoothdown", "fade", "dissolve"], ["smoothleft", "fade", "dissolve"], ["fade", "fadeblack", "dissolve"], hi_mult=0.7, dur=0.45),

    # 51-60: Zoom focused
    _p("Zoom Master", ["zoomin", "circleopen", "radial"], ["zoomin", "fade", "dissolve"], ["fade", "dissolve", "fadewhite"]),
    _p("Zoom and Boom", ["zoomin", "slideleft", "circleopen", "wipeleft"], ["zoomin", "slideleft", "fade"], ["fade", "dissolve"], hi_mult=0.3),
    _p("Zoom Flow", ["zoomin", "circleopen", "fade", "dissolve"], ["zoomin", "fade", "dissolve"], ["fade", "fadeblack", "dissolve"], hi_mult=0.5),
    _p("Zoom Rush", ["zoomin", "slideleft", "slideright", "circleopen", "radial"], ["zoomin", "slideleft", "fade"], ["fade", "dissolve"], hi_mult=0.25, variety=0.8),
    _p("Zoom Pulse", ["zoomin", "circleopen", "circleclose", "radial"], ["zoomin", "fade", "dissolve"], ["fade", "fadewhite", "dissolve"], hi_mult=0.4),
    _p("Zoom and Slide", ["zoomin", "slideleft", "slideright", "slideup"], ["zoomin", "slideleft", "fade"], ["fade", "dissolve"], hi_mult=0.35),
    _p("Zoom and Wipe", ["zoomin", "wipeleft", "wiperight", "circleopen"], ["zoomin", "wipeleft", "fade"], ["fade", "dissolve"], hi_mult=0.3),
    _p("Zoom Galaxy", ["zoomin", "circleopen", "radial", "dissolve", "fade"], ["zoomin", "radial", "fade"], ["fade", "fadewhite", "dissolve"], hi_mult=0.4, variety=0.7),
    _p("Zoom Nova", ["zoomin", "circleopen", "slideleft", "radial"], ["zoomin", "fade", "dissolve"], ["fade", "dissolve", "fadewhite"], hi_mult=0.3, variety=0.8),
    _p("Zoom Eclipse", ["zoomin", "circleopen", "circleclose", "fade", "dissolve"], ["zoomin", "fade", "dissolve"], ["fade", "fadeblack", "dissolve"], hi_mult=0.5),

    # 61-70: Fade focused (minimal)
    _p("Pure Fade", ["fade", "dissolve", "fadewhite"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.8, dur=0.5),
    _p("Fade Deluxe", ["fade", "dissolve", "smoothleft", "fadewhite"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.75, dur=0.5),
    _p("Dissolve Master", ["dissolve", "fade", "radial", "fadewhite"], ["dissolve", "fade", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.7, dur=0.5),
    _p("Fade to Black", ["fade", "fadeblack", "dissolve", "fadewhite"], ["fade", "fadeblack", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.7, dur=0.5),
    _p("White Out", ["fade", "fadewhite", "dissolve", "fadeblack"], ["fade", "fadewhite", "dissolve"], ["fade", "fadewhite", "fadeblack", "dissolve"], hi_mult=0.7, dur=0.5),
    _p("Soft Fade", ["fade", "dissolve", "smoothleft", "smoothright"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "dissolve"], hi_mult=0.85, lo_mult=1.6, dur=0.55),
    _p("Gentle Dissolve", ["dissolve", "fade", "smoothleft", "fadewhite"], ["dissolve", "fade", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.85, lo_mult=1.5, dur=0.55),
    _p("Classic Fade", ["fade", "dissolve", "fadeblack"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.8, dur=0.5),
    _p("Dreamy Fade", ["fade", "fadewhite", "dissolve", "smoothleft", "smoothright"], ["fade", "dissolve", "smoothleft"], ["fade", "fadewhite", "fadeblack", "dissolve"], hi_mult=0.9, lo_mult=1.7, dur=0.6),
    _p("Mellow Fade", ["fade", "dissolve", "fadewhite", "fadeblack"], ["fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.9, lo_mult=1.6, dur=0.55),

    # 71-80: Mixed variety presets
    _p("Chaos Mix", ["zoomin", "slideleft", "wipeleft", "circleopen", "slideright"], ["slideleft", "radial", "fade", "dissolve"], ["fade", "dissolve", "fadewhite"], hi_mult=0.3, variety=1.0),
    _p("Random Stew", ["slideleft", "zoomin", "circleopen", "wipeleft", "wiperight", "slideright"], ["slideleft", "radial", "fade", "dissolve", "smoothleft"], ["fade", "dissolve", "fadewhite", "fadeblack"], hi_mult=0.35, variety=0.95),
    _p("Wild Card", ["zoomin", "circleopen", "circleclose", "slideleft", "slideright", "wipeleft", "wiperight"], ["zoomin", "slideleft", "fade", "dissolve", "radial"], ["fade", "dissolve", "fadewhite"], hi_mult=0.3, variety=1.0),
    _p("Eclectic", ["slideleft", "zoomin", "radial", "circleopen", "wipeleft", "smoothleft"], ["slideleft", "radial", "fade", "dissolve", "smoothleft"], ["fade", "dissolve", "fadeblack"], hi_mult=0.4, variety=0.9),
    _p("Mishmash", ["zoomin", "slideleft", "slideright", "circleopen", "circleclose", "wipeleft", "wiperight", "radial"], ["slideleft", "zoomin", "fade", "dissolve", "radial"], ["fade", "dissolve", "fadewhite", "fadeblack"], hi_mult=0.3, variety=1.0),
    _p("Fusion", ["zoomin", "slideleft", "dissolve", "circleopen", "fade"], ["slideleft", "fade", "dissolve", "smoothleft"], ["fade", "dissolve", "fadeblack"], hi_mult=0.4, variety=0.8),
    _p("Hybrid Flow", ["slideleft", "zoomin", "smoothleft", "circleopen", "fade"], ["slideleft", "smoothleft", "fade", "dissolve"], ["fade", "dissolve", "fadeblack"], hi_mult=0.5, variety=0.7),
    _p("Crossover", ["zoomin", "slideleft", "wipeleft", "dissolve", "circleopen", "fade"], ["slideleft", "dissolve", "fade", "smoothleft"], ["fade", "dissolve", "fadewhite"], hi_mult=0.4, variety=0.8),
    _p("Blend Master", ["slideleft", "zoomin", "circleopen", "dissolve", "fade", "smoothleft", "radial"], ["slideleft", "fade", "dissolve", "smoothleft"], ["fade", "dissolve", "fadeblack", "fadewhite"], hi_mult=0.45, variety=0.85),
    _p("Mix Master", ["zoomin", "slideleft", "slideright", "circleopen", "circleclose", "wipeleft", "wiperight", "radial", "dissolve", "fade"], ["slideleft", "zoomin", "fade", "dissolve", "radial", "smoothleft"], ["fade", "dissolve", "fadewhite", "fadeblack"], hi_mult=0.35, variety=1.0),

    # 81-90: Mood-based presets
    _p("Melancholy", ["dissolve", "fade", "fadeblack", "smoothleft"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.8, lo_mult=1.6, dur=0.55),
    _p("Euphoria", ["zoomin", "slideleft", "circleopen", "wipeleft", "slideright"], ["slideleft", "zoomin", "fade", "dissolve"], ["fade", "dissolve", "fadewhite"], hi_mult=0.3, variety=0.8),
    _p("Mystery", ["dissolve", "fadeblack", "radial", "fade", "dissolve"], ["fade", "dissolve", "fadeblack"], ["fade", "fadeblack", "dissolve", "fadewhite"], hi_mult=0.7, dur=0.5),
    _p("Romance", ["fade", "dissolve", "fadewhite", "smoothleft", "smoothright"], ["fade", "dissolve", "smoothleft"], ["fade", "fadewhite", "dissolve", "fadeblack"], hi_mult=0.85, lo_mult=1.6, dur=0.55),
    _p("Tension", ["zoomin", "slideleft", "wipeleft", "dissolve", "slideright"], ["slideleft", "dissolve", "fade"], ["fade", "dissolve", "fadeblack"], hi_mult=0.4, variety=0.7),
    _p("Triumph", ["zoomin", "circleopen", "slideleft", "radial", "fade"], ["zoomin", "slideleft", "fade", "dissolve"], ["fade", "fadewhite", "dissolve"], hi_mult=0.4, variety=0.7),
    _p("Nostalgia", ["dissolve", "fade", "fadeblack", "fadewhite", "smoothleft"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.8, lo_mult=1.5, dur=0.55),
    _p("Serene", ["fade", "dissolve", "smoothleft", "smoothright", "smoothup", "smoothdown"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.9, lo_mult=1.7, dur=0.6),
    _p("Epic", ["zoomin", "circleopen", "radial", "slideleft", "dissolve", "fade"], ["zoomin", "slideleft", "fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.4, variety=0.6, dur=0.45),
    _p("Intimate", ["fade", "dissolve", "smoothleft", "smoothright", "fadewhite"], ["fade", "dissolve", "smoothleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.9, lo_mult=1.6, dur=0.55),

    # 91-100: Signature combos
    _p("Signature Gold", ["zoomin", "circleopen", "slideleft", "radial", "dissolve"], ["zoomin", "slideleft", "fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.35, variety=0.7, dur=0.4),
    _p("Signature Platinum", ["slideleft", "slideright", "zoomin", "circleopen", "smoothleft"], ["slideleft", "zoomin", "fade", "dissolve", "smoothleft"], ["fade", "dissolve", "fadeblack", "fadewhite"], hi_mult=0.4, variety=0.7, dur=0.4),
    _p("Signature Diamond", ["zoomin", "radial", "circleopen", "dissolve", "fade", "slideleft"], ["zoomin", "radial", "fade", "dissolve"], ["fade", "fadewhite", "dissolve", "fadeblack"], hi_mult=0.35, variety=0.8, dur=0.42),
    _p("Signature Velvet", ["smoothleft", "smoothright", "smoothup", "smoothdown", "fade", "dissolve", "zoomin"], ["smoothleft", "fade", "dissolve", "zoomin"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.5, variety=0.6, dur=0.45),
    _p("Signature Neon", ["zoomin", "slideleft", "circleopen", "wipeleft", "wiperight", "slideright"], ["slideleft", "zoomin", "fade", "dissolve"], ["fade", "dissolve", "fadewhite"], hi_mult=0.3, variety=0.85, dur=0.35),
    _p("Signature Classic", ["fade", "dissolve", "fadeblack", "fadewhite", "zoomin", "slideleft"], ["fade", "dissolve", "slideleft"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.6, variety=0.6, dur=0.45),
    _p("Signature Modern", ["slideleft", "slideright", "zoomin", "circleopen", "radial", "smoothleft"], ["slideleft", "zoomin", "fade", "dissolve", "smoothleft"], ["fade", "dissolve", "fadewhite", "fadeblack"], hi_mult=0.4, variety=0.7, dur=0.4),
    _p("Signature Retro", ["wipeleft", "wiperight", "wipeup", "wipedown", "fade", "dissolve", "fadeblack"], ["wipeleft", "wiperight", "fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.5, variety=0.7, dur=0.4),
    _p("Signature Future", ["zoomin", "circleopen", "circleclose", "radial", "slideleft", "slideright", "dissolve"], ["zoomin", "radial", "slideleft", "fade", "dissolve"], ["fade", "fadewhite", "dissolve", "fadeblack"], hi_mult=0.3, variety=0.85, dur=0.38),
    _p("Signature Royal", ["zoomin", "circleopen", "radial", "dissolve", "fade", "fadeblack", "fadewhite", "slideleft"], ["zoomin", "slideleft", "fade", "dissolve"], ["fade", "fadeblack", "fadewhite", "dissolve"], hi_mult=0.4, variety=0.75, dur=0.42),
]

# Quick lookup
PRESET_BY_NAME: Dict[str, TransitionPreset] = {p.name: p for p in TRANSITION_PRESETS}


def get_preset_names() -> List[str]:
    """Return list of all 100 preset names for UI dropdown."""
    return [p.name for p in TRANSITION_PRESETS]


def get_preset(index: int) -> TransitionPreset:
    """Get preset by index (0-99). Wraps around."""
    return TRANSITION_PRESETS[index % len(TRANSITION_PRESETS)]


def get_preset_by_name(name: str) -> TransitionPreset:
    """Get preset by name. Falls back to first."""
    return PRESET_BY_NAME.get(name, TRANSITION_PRESETS[0])


def get_random_preset(rng=None) -> TransitionPreset:
    """Get a random preset."""
    import random as _r
    r = rng or _r
    return r.choice(TRANSITION_PRESETS)
