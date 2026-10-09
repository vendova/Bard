"""
100 Font Style Combo Presets for lyrics overlay.

Each combo pairs a primary font (for the main lyric line) with a secondary
font (for the preceding/following line or accent text), plus styling
parameters: size, color, outline color, position, alignment, and animation
style.

Fonts are stored in assets/fonts/ and referenced by filename.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")


@dataclass
class FontStyle:
    font_file: str
    size: int = 48
    primary_color: str = "&H00FFFFFF"   # ASS BGR format: &H00BBGGRR
    outline_color: str = "&H00000000"
    outline_width: int = 2
    shadow: int = 1
    bold: bool = False
    italic: bool = False
    alignment: int = 2   # 2 = bottom-center (ASS numpad)
    margin_v: int = 60
    margin_l: int = 40
    margin_r: int = 40

    @property
    def font_path(self) -> str:
        return os.path.join(FONTS_DIR, self.font_file)


@dataclass
class FontCombo:
    name: str
    primary: FontStyle
    secondary: FontStyle
    animation: str = "fade"    # fade, slide_up, slide_left, pop, karaoke, typewriter
    bg_box: bool = False       # semi-transparent background box behind text
    bg_color: str = "&H80000000"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "primary_font": self.primary.font_file,
            "primary_size": self.primary.size,
            "secondary_font": self.secondary.font_file,
            "secondary_size": self.secondary.size,
            "animation": self.animation,
            "bg_box": self.bg_box,
        }


def _c(hex_bgr: str) -> str:
    """Helper to format ASS color."""
    return hex_bgr


# Color presets (ASS format: &H00BBGGRR, &H00 = opaque, &HFF = transparent)
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
YELLOW = "&H0000D7FF"
CYAN = "&H00FFFF00"
MAGENTA = "&H00FF00FF"
ORANGE = "&H000077FF"
GREEN = "&H0000FF00"
RED = "&H000000FF"
PINK = "&H00FFB0C0"
GOLD = "&H0000D4D4"
BLUE = "&H00FF0000"
PURPLE = "&H00C00080"
LIME = "&H0000D000"


def _combo(name, pf, ps, psize=48, ssize=36, pcol=WHITE, scol=YELLOW,
           pbold=False, pital=False, sbold=False, sital=False,
           ow=2, anim="fade", bg=False, align=2, mv=60,
           pcol_outline=BLACK, scol_outline=BLACK):
    return FontCombo(
        name=name,
        primary=FontStyle(font_file=pf, size=psize, primary_color=pcol,
                          outline_color=pcol_outline, outline_width=ow,
                          bold=pbold, italic=pital, alignment=align, margin_v=mv),
        secondary=FontStyle(font_file=ps, size=ssize, primary_color=scol,
                            outline_color=scol_outline, outline_width=ow,
                            bold=sbold, italic=sital, alignment=align, margin_v=mv + ssize + 20),
        animation=anim, bg_box=bg,
    )


# ---------------------------------------------------------------------------
# 100 Font Combos
# ---------------------------------------------------------------------------

FONT_COMBOS: List[FontCombo] = [
    # 1-10: Script + Display combos
    _combo("Dakota Rough + Coldia", "Pacifico-Regular.ttf", "BebasNeue-Regular.ttf", 52, 38, WHITE, CYAN, anim="fade"),
    _combo("Hello Sister Script + Diagram Black Ital", "LoveYaLikeASister.ttf", "JosefinSans-Italic.ttf", 48, 34, WHITE, MAGENTA, pital=False, sital=True, anim="slide_up"),
    _combo("Oi + Prosecco and Baguette", "Oi-Regular.ttf", "Pacifico-Regular.ttf", 50, 36, YELLOW, WHITE, anim="pop"),
    _combo("Pacifico + Bebas Neue", "Pacifico-Regular.ttf", "BebasNeue-Regular.ttf", 54, 40, WHITE, ORANGE, anim="fade"),
    _combo("Lobster + Oswald", "Lobster-Regular.ttf", "Oswald-Regular.ttf", 50, 36, WHITE, CYAN, anim="slide_left"),
    _combo("Caveat + Anton", "Caveat-Regular.ttf", "Anton-Regular.ttf", 52, 38, WHITE, YELLOW, anim="slide_up"),
    _combo("Satisfy + Orbitron", "Satisfy-Regular.ttf", "Orbitron-Regular.ttf", 48, 34, CYAN, WHITE, anim="pop"),
    _combo("Permanent Marker + Audiowide", "PermanentMarker-Regular.ttf", "Audiowide-Regular.ttf", 46, 34, WHITE, MAGENTA, anim="fade"),
    _combo("Kalam + Montserrat", "Kalam-Regular.ttf", "Montserrat-Regular.ttf", 44, 32, WHITE, ORANGE, anim="slide_up"),
    _combo("Oi + Playfair Display", "Oi-Regular.ttf", "PlayfairDisplay-Regular.ttf", 50, 36, GOLD, WHITE, anim="fade"),

    # 11-20: Bold Display combos
    _combo("Anton + Bebas Neue", "Anton-Regular.ttf", "BebasNeue-Regular.ttf", 56, 40, WHITE, YELLOW, pbold=True, anim="pop"),
    _combo("Bebas + Oswald Bold", "BebasNeue-Regular.ttf", "Oswald-Regular.ttf", 52, 38, WHITE, CYAN, pbold=True, anim="slide_left"),
    _combo("Anton + Montserrat", "Anton-Regular.ttf", "Montserrat-Regular.ttf", 54, 36, WHITE, ORANGE, pbold=True, anim="fade"),
    _combo("Orbitron + Audiowide", "Orbitron-Regular.ttf", "Audiowide-Regular.ttf", 48, 34, CYAN, WHITE, anim="slide_up"),
    _combo("Anton + Playfair", "Anton-Regular.ttf", "PlayfairDisplay-Regular.ttf", 52, 36, WHITE, GOLD, pbold=True, anim="fade"),
    _combo("Bebas + Josefin Sans", "BebasNeue-Regular.ttf", "JosefinSans-Regular.ttf", 50, 34, WHITE, MAGENTA, anim="slide_left"),
    _combo("Oswald + Comfortaa", "Oswald-Regular.ttf", "Comfortaa-Regular.ttf", 48, 34, WHITE, CYAN, anim="pop"),
    _combo("Anton + Caveat", "Anton-Regular.ttf", "Caveat-Regular.ttf", 50, 36, YELLOW, WHITE, anim="slide_up"),
    _combo("Orbitron + Bebas", "Orbitron-Regular.ttf", "BebasNeue-Regular.ttf", 46, 38, CYAN, WHITE, anim="fade"),
    _combo("Audiowide + Oswald", "Audiowide-Regular.ttf", "Oswald-Regular.ttf", 44, 34, MAGENTA, WHITE, anim="slide_left"),

    # 21-30: Elegant Serif combos
    _combo("Playfair + Josefin Slab", "PlayfairDisplay-Regular.ttf", "JosefinSlab-Regular.ttf", 48, 34, WHITE, GOLD, anim="fade"),
    _combo("Playfair Ital + Josefin Ital", "PlayfairDisplay-Italic.ttf", "JosefinSans-Italic.ttf", 46, 32, WHITE, PINK, pital=True, sital=True, anim="slide_up"),
    _combo("Playfair + Montserrat Ital", "PlayfairDisplay-Regular.ttf", "Montserrat-Italic.ttf", 48, 34, GOLD, WHITE, sital=True, anim="fade"),
    _combo("Josefin Slab + Playfair Ital", "JosefinSlab-Regular.ttf", "PlayfairDisplay-Italic.ttf", 44, 32, WHITE, CYAN, sital=True, anim="slide_left"),
    _combo("Playfair + Bebas", "PlayfairDisplay-Regular.ttf", "BebasNeue-Regular.ttf", 50, 36, WHITE, ORANGE, anim="pop"),
    _combo("Josefin Slab Ital + Oi", "JosefinSlab-Italic.ttf", "Oi-Regular.ttf", 42, 34, PINK, WHITE, pital=True, anim="fade"),
    _combo("Playfair Ital + Anton", "PlayfairDisplay-Italic.ttf", "Anton-Regular.ttf", 46, 38, WHITE, YELLOW, pital=True, anim="slide_up"),
    _combo("Playfair + Pacifico", "PlayfairDisplay-Regular.ttf", "Pacifico-Regular.ttf", 46, 34, GOLD, WHITE, anim="fade"),
    _combo("Josefin Slab + Satisfy", "JosefinSlab-Regular.ttf", "Satisfy-Regular.ttf", 44, 32, WHITE, CYAN, anim="slide_left"),
    _combo("Playfair + Love Ya Sister", "PlayfairDisplay-Regular.ttf", "LoveYaLikeASister.ttf", 46, 34, WHITE, MAGENTA, anim="pop"),

    # 31-40: Neon/Glow combos (thicker outline)
    _combo("Neon Cyan + Magenta", "Orbitron-Regular.ttf", "Audiowide-Regular.ttf", 48, 34, CYAN, MAGENTA, ow=4, anim="pop", bg=True),
    _combo("Neon Pink + Blue", "Audiowide-Regular.ttf", "Orbitron-Regular.ttf", 46, 34, PINK, BLUE, ow=4, anim="fade", bg=True),
    _combo("Neon Green + Yellow", "Orbitron-Regular.ttf", "BebasNeue-Regular.ttf", 48, 38, GREEN, YELLOW, ow=3, anim="slide_up", bg=True),
    _combo("Neon Gold + Purple", "Audiowide-Regular.ttf", "Montserrat-Regular.ttf", 44, 32, GOLD, PURPLE, ow=4, anim="slide_left", bg=True),
    _combo("Neon Orange + Cyan", "BebasNeue-Regular.ttf", "Orbitron-Regular.ttf", 50, 36, ORANGE, CYAN, ow=3, anim="pop", bg=True),
    _combo("Neon Red + White", "Anton-Regular.ttf", "Oswald-Regular.ttf", 52, 36, RED, WHITE, ow=4, anim="fade", bg=True),
    _combo("Neon Lime + Magenta", "Orbitron-Regular.ttf", "Audiowide-Regular.ttf", 46, 34, LIME, MAGENTA, ow=3, anim="slide_up", bg=True),
    _combo("Neon Blue + Gold", "Audiowide-Regular.ttf", "BebasNeue-Regular.ttf", 44, 36, BLUE, GOLD, ow=4, anim="slide_left", bg=True),
    _combo("Neon Purple + Pink", "Montserrat-Regular.ttf", "Orbitron-Regular.ttf", 42, 34, PURPLE, PINK, ow=3, anim="pop", bg=True),
    _combo("Neon White + Cyan", "BebasNeue-Regular.ttf", "Audiowide-Regular.ttf", 48, 34, WHITE, CYAN, ow=4, anim="fade", bg=True),

    # 41-50: Handwritten combos
    _combo("Caveat + Kalam", "Caveat-Regular.ttf", "Kalam-Regular.ttf", 50, 34, WHITE, ORANGE, anim="slide_up"),
    _combo("Kalam + Satisfy", "Kalam-Regular.ttf", "Satisfy-Regular.ttf", 44, 32, WHITE, CYAN, anim="fade"),
    _combo("Satisfy + Permanent Marker", "Satisfy-Regular.ttf", "PermanentMarker-Regular.ttf", 46, 34, WHITE, MAGENTA, anim="slide_left"),
    _combo("Permanent Marker + Caveat", "PermanentMarker-Regular.ttf", "Caveat-Regular.ttf", 44, 32, YELLOW, WHITE, anim="pop"),
    _combo("Love Ya Sister + Kalam", "LoveYaLikeASister.ttf", "Kalam-Regular.ttf", 46, 32, WHITE, PINK, anim="fade"),
    _combo("Caveat + Pacifico", "Caveat-Regular.ttf", "Pacifico-Regular.ttf", 46, 34, WHITE, CYAN, anim="slide_up"),
    _combo("Kalam + Satisfy Ital", "Kalam-Regular.ttf", "Satisfy-Regular.ttf", 44, 32, ORANGE, WHITE, anim="slide_left"),
    _combo("Permanent Marker + Oi", "PermanentMarker-Regular.ttf", "Oi-Regular.ttf", 44, 34, WHITE, YELLOW, anim="pop"),
    _combo("Satisfy + Lobster", "Satisfy-Regular.ttf", "Lobster-Regular.ttf", 46, 34, WHITE, GOLD, anim="fade"),
    _combo("Caveat + Love Ya Sister", "Caveat-Regular.ttf", "LoveYaLikeASister.ttf", 44, 32, CYAN, WHITE, anim="slide_up"),

    # 51-60: Minimalist combos
    _combo("Montserrat + Josefin", "Montserrat-Regular.ttf", "JosefinSans-Regular.ttf", 44, 32, WHITE, CYAN, anim="fade"),
    _combo("Josefin + Montserrat Ital", "JosefinSans-Regular.ttf", "Montserrat-Italic.ttf", 42, 32, WHITE, ORANGE, sital=True, anim="slide_left"),
    _combo("Montserrat + Comfortaa", "Montserrat-Regular.ttf", "Comfortaa-Regular.ttf", 44, 32, WHITE, PINK, anim="pop"),
    _combo("Comfortaa + Josefin", "Comfortaa-Regular.ttf", "JosefinSans-Regular.ttf", 42, 30, WHITE, CYAN, anim="fade"),
    _combo("Montserrat Ital + Josefin Ital", "Montserrat-Italic.ttf", "JosefinSans-Italic.ttf", 42, 30, WHITE, MAGENTA, pital=True, sital=True, anim="slide_up"),
    _combo("Josefin + Comfortaa", "JosefinSans-Regular.ttf", "Comfortaa-Regular.ttf", 42, 30, WHITE, GOLD, anim="slide_left"),
    _combo("Montserrat + Oswald", "Montserrat-Regular.ttf", "Oswald-Regular.ttf", 44, 32, WHITE, ORANGE, anim="fade"),
    _combo("Comfortaa + Montserrat Ital", "Comfortaa-Regular.ttf", "Montserrat-Italic.ttf", 42, 30, CYAN, WHITE, sital=True, anim="pop"),
    _combo("Josefin Ital + Montserrat", "JosefinSans-Italic.ttf", "Montserrat-Regular.ttf", 42, 32, PINK, WHITE, pital=True, anim="slide_up"),
    _combo("Montserrat + Bebas", "Montserrat-Regular.ttf", "BebasNeue-Regular.ttf", 44, 34, WHITE, YELLOW, anim="fade"),

    # 61-70: High-contrast combos
    _combo("Anton + Playfair Ital", "Anton-Regular.ttf", "PlayfairDisplay-Italic.ttf", 52, 34, WHITE, GOLD, sital=True, anim="pop"),
    _combo("Bebas + Caveat", "BebasNeue-Regular.ttf", "Caveat-Regular.ttf", 50, 34, YELLOW, WHITE, anim="slide_up"),
    _combo("Oswald + Pacifico", "Oswald-Regular.ttf", "Pacifico-Regular.ttf", 48, 34, WHITE, CYAN, anim="fade"),
    _combo("Anton + Satisfy", "Anton-Regular.ttf", "Satisfy-Regular.ttf", 50, 34, WHITE, MAGENTA, anim="slide_left"),
    _combo("Bebas + Oi", "BebasNeue-Regular.ttf", "Oi-Regular.ttf", 48, 34, ORANGE, WHITE, anim="pop"),
    _combo("Oswald + Lobster", "Oswald-Regular.ttf", "Lobster-Regular.ttf", 46, 34, WHITE, GOLD, anim="fade"),
    _combo("Anton + Love Ya Sister", "Anton-Regular.ttf", "LoveYaLikeASister.ttf", 50, 34, WHITE, PINK, anim="slide_up"),
    _combo("Bebas + Permanent Marker", "BebasNeue-Regular.ttf", "PermanentMarker-Regular.ttf", 48, 32, CYAN, WHITE, anim="slide_left"),
    _combo("Oswald + Kalam", "Oswald-Regular.ttf", "Kalam-Regular.ttf", 46, 32, WHITE, ORANGE, anim="pop"),
    _combo("Anton + Croissant One", "Anton-Regular.ttf", "CroissantOne-Regular.ttf", 50, 34, YELLOW, WHITE, anim="fade"),

    # 71-80: Color pop combos
    _combo("Yellow on Black + Cyan", "Anton-Regular.ttf", "BebasNeue-Regular.ttf", 52, 38, YELLOW, CYAN, ow=3, anim="pop", bg=True),
    _combo("Pink + Gold", "Pacifico-Regular.ttf", "Lobster-Regular.ttf", 48, 34, PINK, GOLD, anim="fade"),
    _combo("Orange + Blue", "Oi-Regular.ttf", "Orbitron-Regular.ttf", 46, 34, ORANGE, BLUE, ow=3, anim="slide_up"),
    _combo("Green + Magenta", "Audiowide-Regular.ttf", "Montserrat-Regular.ttf", 44, 32, GREEN, MAGENTA, ow=3, anim="slide_left"),
    _combo("Gold + Purple", "PlayfairDisplay-Regular.ttf", "JosefinSlab-Regular.ttf", 46, 32, GOLD, PURPLE, ow=3, anim="pop"),
    _combo("Cyan + Red", "BebasNeue-Regular.ttf", "Oswald-Regular.ttf", 50, 36, CYAN, RED, ow=3, anim="fade"),
    _combo("Lime + Pink", "Orbitron-Regular.ttf", "Comfortaa-Regular.ttf", 44, 32, LIME, PINK, ow=3, anim="slide_up"),
    _combo("White + Orange Box", "Montserrat-Regular.ttf", "JosefinSans-Regular.ttf", 44, 32, WHITE, ORANGE, ow=2, anim="fade", bg=True),
    _combo("Magenta + Gold", "LoveYaLikeASister.ttf", "Oi-Regular.ttf", 46, 34, MAGENTA, GOLD, ow=3, anim="pop"),
    _combo("Blue + Yellow", "Audiowide-Regular.ttf", "BebasNeue-Regular.ttf", 44, 36, BLUE, YELLOW, ow=3, anim="slide_left"),

    # 81-90: Mixed style combos
    _combo("Script Bold + Tech", "Pacifico-Regular.ttf", "Orbitron-Regular.ttf", 48, 34, WHITE, CYAN, anim="fade"),
    _combo("Marker + Elegant", "PermanentMarker-Regular.ttf", "PlayfairDisplay-Italic.ttf", 44, 32, WHITE, GOLD, sital=True, anim="slide_up"),
    _combo("Handwritten + Display", "Caveat-Regular.ttf", "Anton-Regular.ttf", 46, 36, WHITE, YELLOW, anim="pop"),
    _combo("Elegant + Tech", "PlayfairDisplay-Regular.ttf", "Audiowide-Regular.ttf", 46, 34, GOLD, WHITE, anim="slide_left"),
    _combo("Tech + Script", "Orbitron-Regular.ttf", "Satisfy-Regular.ttf", 44, 32, CYAN, WHITE, anim="fade"),
    _combo("Display + Handwritten", "BebasNeue-Regular.ttf", "Kalam-Regular.ttf", 48, 32, WHITE, ORANGE, anim="slide_up"),
    _combo("Script + Slab", "Lobster-Regular.ttf", "JosefinSlab-Regular.ttf", 46, 32, WHITE, CYAN, anim="pop"),
    _combo("Slab + Script", "JosefinSlab-Regular.ttf", "Pacifico-Regular.ttf", 44, 32, WHITE, MAGENTA, anim="fade"),
    _combo("Sans + Display Ital", "Montserrat-Regular.ttf", "PlayfairDisplay-Italic.ttf", 44, 32, WHITE, PINK, sital=True, anim="slide_left"),
    _combo("Display + Mono Ital", "Anton-Regular.ttf", "JosefinSans-Italic.ttf", 48, 32, WHITE, ORANGE, sital=True, anim="pop"),

    # 91-100: Unique creative combos
    _combo("Croissant + Oi", "CroissantOne-Regular.ttf", "Oi-Regular.ttf", 46, 34, GOLD, WHITE, anim="fade"),
    _combo("Patua + Pacifico", "PatuaOne-Regular.ttf", "Pacifico-Regular.ttf", 46, 34, WHITE, CYAN, anim="slide_up"),
    _combo("Oi + Caveat", "Oi-Regular.ttf", "Caveat-Regular.ttf", 48, 32, YELLOW, WHITE, anim="slide_left"),
    _combo("Croissant + Bebas", "CroissantOne-Regular.ttf", "BebasNeue-Regular.ttf", 46, 36, GOLD, WHITE, anim="pop"),
    _combo("Patua + Oi", "PatuaOne-Regular.ttf", "Oi-Regular.ttf", 44, 34, WHITE, ORANGE, anim="fade"),
    _combo("Lobster Two + Montserrat", "Lobster-Regular.ttf", "Montserrat-Regular.ttf", 46, 32, WHITE, CYAN, anim="slide_up"),
    _combo("Pacifico + Josefin Ital", "Pacifico-Regular.ttf", "JosefinSans-Italic.ttf", 46, 32, WHITE, PINK, sital=True, anim="slide_left"),
    _combo("Satisfy + Playfair", "Satisfy-Regular.ttf", "PlayfairDisplay-Regular.ttf", 44, 32, CYAN, WHITE, anim="pop"),
    _combo("Oi + Anton", "Oi-Regular.ttf", "Anton-Regular.ttf", 48, 36, MAGENTA, WHITE, anim="fade"),
    _combo("Love Ya + Bebas", "LoveYaLikeASister.ttf", "BebasNeue-Regular.ttf", 46, 36, WHITE, YELLOW, anim="slide_up"),
]

# Quick lookup by name
COMBO_BY_NAME: Dict[str, FontCombo] = {c.name: c for c in FONT_COMBOS}


def get_combo_names() -> List[str]:
    """Return list of all 100 combo names for UI dropdown."""
    return [c.name for c in FONT_COMBOS]


def get_combo(index: int) -> FontCombo:
    """Get combo by index (0-99). Wraps around."""
    return FONT_COMBOS[index % len(FONT_COMBOS)]


def get_combo_by_name(name: str) -> FontCombo:
    """Get combo by name. Falls back to first combo."""
    return COMBO_BY_NAME.get(name, FONT_COMBOS[0])


def get_random_combo(rng=None) -> FontCombo:
    """Get a random combo."""
    import random as _r
    r = rng or _r
    return r.choice(FONT_COMBOS)


def validate_fonts() -> List[str]:
    """Check which font files are missing. Returns list of missing filenames."""
    missing = []
    for combo in FONT_COMBOS:
        for style in (combo.primary, combo.secondary):
            if not os.path.exists(style.font_path):
                if style.font_file not in missing:
                    missing.append(style.font_file)
    return missing
