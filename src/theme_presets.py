"""
Theme Presets — one choice that bundles the whole visual identity.

A theme binds a font combo (lyrics), a cinematic look (colour grade), a
transition preset (Smart Sync cuts) and a small set of per-clip effect
defaults. Selecting a theme fills the other dropdowns/toggles from a single
name, which is convenient for quickly trying a coherent style.

Mirrors the ``FontCombo`` / ``TransitionPreset`` / ``LookPreset`` pattern.
The resolved values are stored on the render (draft manifest + job settings)
so a re-render reproduces the same theme.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ThemePreset:
    name: str
    description: str = ""

    # Bindings into the other preset families (names must exist there).
    font_combo: str = ""
    look: str = ""
    transition_preset: str = ""

    # Effect defaults the theme enables.
    effects_enabled: bool = True
    vignette: bool = False
    vignette_strength: float = 0.4
    color_boost: bool = False
    color_boost_amount: float = 1.3
    gradient_overlay: bool = False

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "font_combo": self.font_combo,
            "look": self.look,
            "transition_preset": self.transition_preset,
            "effects_enabled": self.effects_enabled,
        }


def _t(name, desc="", **kw) -> ThemePreset:
    return ThemePreset(name=name, description=desc, **kw)


THEME_PRESETS: List[ThemePreset] = [
    _t("None", "Keep my manual font / look / transition / effects picks"),
    _t("Cinematic Blockbuster", "Teal & orange grade, bold sans, modern cuts",
       font_combo="Anton + Bebas Neue", look="Blockbuster",
       transition_preset="Signature Modern", effects_enabled=True,
       vignette=True, vignette_strength=0.35),
    _t("Neo Noir", "Cold neon noir grade, condensed type, dissolve cuts",
       font_combo="Bebas + Oswald Bold", look="Neo Noir Night",
       transition_preset="Mystery", effects_enabled=True,
       vignette=True, vignette_strength=0.45),
    _t("Vintage Film", "Faded warm film, serif type, soft fades",
       font_combo="Playfair + Josefin Slab", look="Warm Vintage",
       transition_preset="Nostalgia", effects_enabled=True,
       vignette=True, vignette_strength=0.5),
    _t("Golden Pop", "Warm golden grade, playful type, upbeat fades",
       font_combo="Pacifico + Bebas Neue", look="Golden Hour",
       transition_preset="Euphoria", effects_enabled=True,
       color_boost=True, color_boost_amount=1.2),
    _t("Neon Cyber", "Magenta/cyan grade, techno type, glitch energy",
       font_combo="Orbitron + Audiowide", look="Cyberpunk",
       transition_preset="Signature Neon", effects_enabled=True,
       color_boost=True, color_boost_amount=1.35),
    _t("Monochrome Noir", "Pure B&W noir grade, classic type, slow fades",
       font_combo="Bebas + Oswald Bold", look="Noir",
       transition_preset="Classic Fade", effects_enabled=True,
       vignette=True, vignette_strength=0.55),
    _t("Dreamy Pastel", "Airy pastel grade, soft script, gentle fades",
       font_combo="Josefin Slab + Satisfy", look="Pastel Dream",
       transition_preset="Dreamy Fade", effects_enabled=True,
       vignette=True, vignette_strength=0.3),
]

THEME_BY_NAME: Dict[str, ThemePreset] = {t.name: t for t in THEME_PRESETS}


def get_theme_names() -> List[str]:
    return [t.name for t in THEME_PRESETS]


def get_theme(index: int) -> ThemePreset:
    return THEME_PRESETS[index % len(THEME_PRESETS)]


def get_theme_by_name(name: str) -> ThemePreset:
    return THEME_BY_NAME.get(name, THEME_PRESETS[0])


def is_noop(theme: Optional[ThemePreset]) -> bool:
    """True for the 'None' placeholder (or nothing) — leave manual picks alone."""
    return theme is None or theme.name == "None"


def get_random_theme(rng=None) -> ThemePreset:
    """Random non-'None' theme."""
    r = rng or random
    choices = [t for t in THEME_PRESETS if t.name != "None"] or THEME_PRESETS
    return r.choice(choices)
