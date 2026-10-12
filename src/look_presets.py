"""
Cinematic Look Presets (colour grading).

Each preset is a token table that maps a named look to FFmpeg-native grade
filters: ``colorbalance``/``eq``/``vibrance`` for the base grade, ``curves``
for the tone curve, ``noise`` for film grain, and a ``split``/``gblur``/
``blend=screen`` sub-graph for halation (light bloom). A ``lut3d`` is applied
when a matching ``.cube`` sits in ``assets/luts/``.

Mirrors the existing ``FontCombo`` / ``TransitionPreset`` pattern: a dataclass
list, name lookup helpers, and a ``get_random_look`` for the 🎲 Random option.

Everything here is deliberately FFmpeg-native so it stays CPU-only and
portable. Each filter is probed with ``post_ops.has_filter`` and silently
dropped when the local FFmpeg build lacks it (halation needs ``blend`` with
``all_mode=screen``, grain needs ``noise``, etc.).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
LUTS_DIR = os.path.join(ASSETS_DIR, "luts")


@dataclass
class LookPreset:
    """A named colour grade expressed as FFmpeg filter parameters."""

    name: str
    description: str = ""

    # colorbalance — per-range RGB shifts, each in [-1, 1].
    cb_shadows: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cb_midtones: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cb_highlights: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # eq / vibrance
    contrast: float = 1.0
    saturation: float = 1.0
    brightness: float = 0.0
    gamma: float = 1.0
    vibrance: float = 0.0

    # film grain (noise filter strength; 0 = off)
    grain: int = 0

    # halation / bloom (0 = off): blurred highlights screened back over the frame
    halation: float = 0.0
    halation_blur: float = 8.0

    # tone curve (FFmpeg curves master control points; "" = off)
    curves: str = ""

    # cinematic letterbox: fraction of height per bar (0 = off, changes framing)
    letterbox: float = 0.0

    # optional .cube LUT filename inside assets/luts/
    lut: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "grain": self.grain,
            "halation": self.halation,
            "letterbox": self.letterbox,
            "lut": self.lut,
        }


def resolve_lut(name: str) -> Optional[str]:
    """Return the path to an installed .cube LUT, or None."""
    if not name:
        return None
    path = name if os.path.isabs(name) else os.path.join(LUTS_DIR, name)
    return path if os.path.isfile(path) else None


def _lut_names() -> List[str]:
    if not os.path.isdir(LUTS_DIR):
        return []
    return sorted(f for f in os.listdir(LUTS_DIR) if f.lower().endswith(".cube"))


def build_look_filters(
    look: "LookPreset",
    target_size: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Translate a look into FFmpeg ``-vf`` filter strings.

    Filters the local build does not support are skipped, and a missing LUT is
    ignored, so this never returns an unusable chain.
    """
    if look is None:
        return []
    from post_ops import has_filter

    filters: List[str] = []
    w, h = (target_size or (1920, 1080))

    # --- Base grade ---
    if has_filter("colorbalance") and any(
        any(abs(v) > 1e-6 for v in t)
        for t in (look.cb_shadows, look.cb_midtones, look.cb_highlights)
    ):
        rs, gs, bs = look.cb_shadows
        rm, gm, bm = look.cb_midtones
        rh, gh, bh = look.cb_highlights
        filters.append(
            f"colorbalance=rs={rs:.3f}:gs={gs:.3f}:bs={bs:.3f}"
            f":rm={rm:.3f}:gm={gm:.3f}:bm={bm:.3f}"
            f":rh={rh:.3f}:gh={gh:.3f}:bh={bh:.3f}"
        )

    if has_filter("eq") and (
        abs(look.contrast - 1.0) > 1e-6 or abs(look.saturation - 1.0) > 1e-6
        or abs(look.brightness) > 1e-6 or abs(look.gamma - 1.0) > 1e-6
    ):
        filters.append(
            f"eq=contrast={look.contrast:.3f}:saturation={look.saturation:.3f}"
            f":brightness={look.brightness:.3f}:gamma={look.gamma:.3f}"
        )

    if look.vibrance and has_filter("vibrance"):
        filters.append(f"vibrance=intensity={look.vibrance:.3f}")

    if look.curves and has_filter("curves"):
        filters.append(f"curves=master='{look.curves}'")

    # --- Optional LUT (assets/luts/*.cube) ---
    lut_path = resolve_lut(look.lut)
    if lut_path and has_filter("lut3d"):
        filters.append(f"lut3d='{lut_path}'")

    # --- Halation (bloom): self-contained split/blend sub-graph that screens
    #     blurred highlights back over the frame. Runs after the colour grade
    #     and before the letterbox pad so bars stay pure black. ---
    if look.halation > 0 and has_filter("blend") and has_filter("gblur"):
        opacity = max(0.05, min(0.9, float(look.halation)))
        filters.append(
            f"split[hal_a][hal_b];[hal_b]gblur=sigma={look.halation_blur:.1f}:steps=1"
            f"[hal_blur];[hal_a][hal_blur]blend=all_mode=screen:all_opacity={opacity:.2f}"
        )

    # --- Film grain (last of the grade so it sits over the colour) ---
    if look.grain and has_filter("noise"):
        filters.append(f"noise=alls={int(look.grain)}:allf=t+u")

    # --- Letterbox bars (changes framing; off unless the look asks for it) ---
    if look.letterbox and look.letterbox > 0:
        bar = int(round(h * float(look.letterbox)))
        bar -= bar % 2  # keep output height even for yuv420p
        if bar > 0:
            filters.append(f"pad=iw:ih+{2 * bar}:0:{bar}:black")

    return filters


# ---------------------------------------------------------------------------
# Look definitions
# ---------------------------------------------------------------------------

def _look(name, desc="", **kw) -> LookPreset:
    return LookPreset(name=name, description=desc, **kw)


LOOK_PRESETS: List[LookPreset] = [
    _look("None", "No grading (pass-through)",
          contrast=1.0, saturation=1.0),
    _look("Teal & Orange", "Blockbuster teal shadows / warm skin",
          cb_shadows=(-0.06, 0.0, 0.09), cb_highlights=(0.09, 0.03, -0.09),
          cb_midtones=(0.04, 0.0, -0.03), contrast=1.10, saturation=1.12,
          vibrance=0.12),
    _look("Warm Vintage", "Faded warm film with soft highlights",
          cb_shadows=(0.05, 0.02, -0.02), cb_highlights=(0.04, 0.02, -0.06),
          contrast=0.96, saturation=0.9, brightness=0.02, gamma=1.04,
          grain=8, curves="0/0.05 0.5/0.5 1/0.94"),
    _look("Bleach Bypass", "Desaturated, high-contrast silver look",
          contrast=1.32, saturation=0.62, brightness=0.01, gamma=0.95,
          grain=6),
    _look("Noir", "High-contrast black & white",
          saturation=0.0, contrast=1.30, gamma=0.98, grain=10,
          curves="0/0 0.25/0.18 0.5/0.5 1/1"),
    _look("Golden Hour", "Warm sunlit glow",
          cb_highlights=(0.10, 0.05, -0.10), cb_midtones=(0.05, 0.02, -0.05),
          contrast=1.05, saturation=1.15, brightness=0.02, gamma=1.02,
          vibrance=0.1),
    _look("Cool Cinema", "Cool blue shadows, neutral highlights",
          cb_shadows=(-0.07, -0.02, 0.10), cb_highlights=(0.0, 0.0, 0.05),
          contrast=1.08, saturation=1.02),
    _look("Cyberpunk", "Magenta/cyan neon with crushed blacks",
          cb_shadows=(-0.05, -0.08, 0.12), cb_highlights=(0.12, -0.04, 0.06),
          cb_midtones=(0.03, -0.02, 0.06), contrast=1.18, saturation=1.25,
          vibrance=0.15, halation=0.22, halation_blur=10.0, grain=4),
    _look("Pastel Dream", "Lifted, airy pastel wash",
          cb_shadows=(0.06, 0.04, 0.06), contrast=0.94, saturation=0.92,
          brightness=0.04, gamma=1.06, curves="0/0.08 0.5/0.52 1/0.96"),
    _look("Sepia", "Classic warm monochrome",
          cb_midtones=(0.12, 0.05, -0.10), saturation=0.25, contrast=1.05,
          grain=7),
    _look("High Contrast", "Punchy contrast, clean colour",
          contrast=1.28, saturation=1.10, gamma=0.96),
    _look("Faded Film", "Low-contrast matte film emulation",
          contrast=0.90, saturation=0.95, brightness=0.03, gamma=1.05,
          grain=5, curves="0/0.06 0.5/0.5 1/0.95"),
    _look("Infrared", "Surreal shifted-flora false colour",
          cb_shadows=(0.05, -0.06, 0.02), cb_highlights=(0.10, 0.06, -0.10),
          saturation=1.2, contrast=1.1, vibrance=0.2),
    _look("Moonlight", "Cold desaturated night",
          cb_shadows=(-0.08, -0.04, 0.12), contrast=1.12, saturation=0.78,
          brightness=-0.02, gamma=0.98),
    _look("Blockbuster", "Cinematic teal/orange with bloom + bars",
          cb_shadows=(-0.07, 0.0, 0.10), cb_highlights=(0.10, 0.04, -0.10),
          contrast=1.12, saturation=1.15, vibrance=0.12,
          halation=0.28, halation_blur=9.0, letterbox=0.055),
    _look("Neo Noir Night", "Neon noir: cold shadows, magenta glow",
          cb_shadows=(-0.09, -0.05, 0.13), cb_highlights=(0.10, -0.03, 0.07),
          contrast=1.16, saturation=1.05, gamma=0.96,
          halation=0.30, halation_blur=12.0, grain=8, letterbox=0.055),
]

LOOK_BY_NAME: Dict[str, LookPreset] = {l.name: l for l in LOOK_PRESETS}


def get_look_names() -> List[str]:
    """Return all look names for the UI dropdown."""
    return [l.name for l in LOOK_PRESETS]


def get_look(index: int) -> LookPreset:
    return LOOK_PRESETS[index % len(LOOK_PRESETS)]


def get_look_by_name(name: str) -> LookPreset:
    return LOOK_BY_NAME.get(name, LOOK_PRESETS[0])


def get_random_look(rng=None) -> LookPreset:
    """Random look, excluding the neutral 'None' placeholder."""
    r = rng or random
    choices = [l for l in LOOK_PRESETS if l.name != "None"] or LOOK_PRESETS
    return r.choice(choices)
