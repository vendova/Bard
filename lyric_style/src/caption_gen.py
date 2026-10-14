"""Caption generator — two-layer lyric lockup as transparent PNG frames.

Layer 1: calligraphic script word (white, oversized swash initial, tilted CCW).
Layer 2: bold condensed all-caps key word (yellow, italic lean, tracked).

Rendered per frame with the Section 6.6 animation (script wipe, key scale,
exit fade+drift). FFmpeg then overlays the PNG sequence on the shot.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(path, size)
    return FONT_CACHE[key]


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _size_for_cap(font_path: str, cap_px: float, probe: str = "H") -> int:
    """Font size whose uppercase cap height is ~cap_px."""
    size = max(8, int(cap_px * 2))
    for _ in range(24):
        f = _font(font_path, size)
        bbox = f.getbbox(probe)
        h = bbox[3] - bbox[1] if bbox else 0
        if h <= 0:
            break
        if abs(h - cap_px) <= max(1, cap_px * 0.02):
            return size
        size = int(size * cap_px / h)
        size = max(8, min(1200, size))
    return size


def _draw_tracked(draw, xy, text, font, fill, tracking):
    """Draw text with extra per-glyph tracking; return total width."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        w = font.getlength(ch)
        x += w + tracking
    return x - tracking - xy[0]


def _text_width(font, text, tracking) -> float:
    return font.getlength(text) + tracking * max(0, len(text) - 1)


def _shear(img: Image.Image, deg: float) -> Image.Image:
    """Italic lean via horizontal shear."""
    if abs(deg) < 0.1:
        return img
    tan = math.tan(math.radians(deg))
    w, h = img.size
    new_w = int(w + abs(tan) * h) + 2
    out = Image.new("RGBA", (new_w, h), (0, 0, 0, 0))
    out.paste(img, (int(abs(tan) * h) if tan < 0 else 0, 0), img)
    return out.transform(
        (new_w, h), Image.AFFINE, (1, tan, -tan * h if tan > 0 else 0, 0, 1, 0),
        resample=Image.BICUBIC,
    )


def _rot(img: Image.Image, deg: float) -> Image.Image:
    return img if abs(deg) < 0.1 else img.rotate(deg, expand=True,
                                                resample=Image.BICUBIC)


def _layer_text(text, font, fill, tracking, shadow, skew_deg=0.0, rot_deg=0.0):
    """Render one text layer (fill + drop shadow) to an RGBA image."""
    pad = 40
    tmp = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = int(_text_width(font, text, tracking)) + pad * 2
    h = (bbox[3] - bbox[1]) + pad * 2
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    _draw_tracked(dl, (pad - bbox[0], pad - bbox[1]), text, font, fill + (255,),
                  tracking)
    if shadow and shadow.get("enabled", True):
        s = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ds = ImageDraw.Draw(s)
        sc = _hex(shadow["color"]) + (int(255 * shadow["opacity"]),)
        _draw_tracked(ds, (pad - bbox[0] + shadow["dx"],
                           pad - bbox[1] + shadow["dy"]), text, font, sc, tracking)
        s = s.filter(ImageFilter.GaussianBlur(shadow["blur"]))
        layer = Image.alpha_composite(s, layer)
    if skew_deg:
        layer = _shear(layer, skew_deg)
    if rot_deg:
        layer = _rot(layer, rot_deg)
    return layer


class CaptionRenderer:
    def __init__(self, cfg: dict, fonts: dict[str, Path], width: int, height: int,
                 vertical: bool = False):
        self.cfg = cfg
        self.cap = cfg["caption"]
        self.vert = cfg["caption_vertical"]
        self.W, self.H = width, height
        self.vertical = vertical
        self.fonts = fonts

    def _metrics(self):
        c = self.cap
        key_cap_frac = (self.vert["key_cap_height_frac"] if self.vertical
                        else c["key_cap_height_frac"])
        key_cap_px = key_cap_frac * self.H
        script_initial_px = key_cap_px * c["script_initial_ratio"]
        return key_cap_px, script_initial_px

    def build_lockup(self, caption: dict) -> tuple[Image.Image, Image.Image]:
        c = self.cap
        key_cap_px, script_initial_px = self._metrics()
        key_font_path = str(self.fonts[caption["key_font"]])
        script_font_path = str(self.fonts[caption["script_font"]])

        key_size = _size_for_cap(key_font_path, key_cap_px, probe="H")
        key_font = _font(key_font_path, key_size)
        key_img = _layer_text(caption["key"], key_font, _hex(c["key_fill"]),
                              c["key_tracking"], {**c["key_shadow"], "enabled": True},
                              skew_deg=c["key_tilt_deg"])

        script_size = _size_for_cap(script_font_path, script_initial_px, probe="H")
        script_font = _font(script_font_path, script_size)
        script_img = _layer_text(caption["script"], script_font,
                                 _hex(c["script_fill"]), c["script_tracking"],
                                 {**c["script_shadow"], "enabled": True},
                                 rot_deg=c["script_tilt_deg"])
        return script_img, key_img

    def compose_static(self, caption: dict) -> Image.Image:
        """Full-frame RGBA lockup at its resting position (fully visible)."""
        c = self.cap
        script_img, key_img = self.build_lockup(caption)
        # Stack: script on top line, key tucked under, overlapping the tail.
        gap = int(0.10 * key_img.height)
        lw = max(script_img.width, key_img.width)
        lh = script_img.height + key_img.height - gap
        lock = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        # Key under script: draw key first (lower), then script (upper).
        lock.alpha_composite(key_img, ((lw - key_img.width) // 2,
                                       script_img.height - gap))
        lock.alpha_composite(script_img, ((lw - script_img.width) // 2, 0))

        frame = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        anchor = caption.get("anchor", "right")
        if self.vertical:
            y_lo, y_hi = self.vert["y_range_frac"]
        else:
            y_lo = self.cap["margin_bottom_frac"][0]
            y_hi = self.cap["margin_bottom_frac"][1]
        m_edge = int(np.random.default_rng(0).uniform(*c["margin_edge_frac"]) * self.W)
        m_bottom = int(0.5 * (y_lo + y_hi) * self.H)
        y = self.H - m_bottom - lock.height
        y = max(int(c["safe_margin_frac"] * self.H), y)
        if self.vertical:
            y = int(min(y, self.H * (1 - self.vert["headroom_frac"]) - lock.height))
            y = max(y, int(self.H * self.vert["y_range_frac"][0]))
        if anchor == "right":
            x = self.W - m_edge - lock.width
        else:
            x = m_edge
        x = max(int(c["safe_margin_frac"] * self.W),
                min(x, self.W - lock.width - int(c["safe_margin_frac"] * self.W)))
        frame.alpha_composite(lock, (x, y))
        return frame

    def frames_for_shot(self, caption: dict, shot_frames: int,
                        cut_is_first: bool = False) -> list[Image.Image]:
        """Return one full-frame RGBA image per shot frame with the animation."""
        base = self.compose_static(caption)
        a = self.cap["anim"]
        out = []
        exit_start = shot_frames - a["exit_frames"] - a["exit_before_cut_frames"]
        for f in range(shot_frames):
            img = base.copy()
            alpha = 1.0
            dy = 0
            # Entrance: nothing before script start; ramp to fully visible.
            if f < a["script_start_frame"]:
                img.putalpha(Image.new("L", img.size, 0))
                out.append(img)
                continue
            # Script reveal fraction (wipe from left).
            if f < a["script_start_frame"] + a["script_in_frames"]:
                frac = (f - a["script_start_frame"]) / a["script_in_frames"]
                img = self._wipe(img, frac)
            # Key word appears a couple frames later; scale-in handled by wipe too.
            if f < a["key_start_frame"]:
                img = self._dim_top(img)
            # Exit: fade + downward drift.
            if f >= exit_start:
                k = (f - exit_start) / max(1, a["exit_frames"])
                alpha = max(0.0, 1.0 - k)
                dy = int(0.01 * self.H * k)
            if alpha < 1.0 or dy:
                img = self._apply_alpha_shift(img, alpha, dy)
            out.append(img)
        return out

    # -- animation helpers -------------------------------------------------- #
    def _wipe(self, img: Image.Image, frac: float) -> Image.Image:
        frac = max(0.0, min(1.0, frac))
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        edge = max(2, int(w * 0.03))
        reveal = int(w * frac)
        md = ImageDraw.Draw(mask)
        if reveal > 0:
            md.rectangle([0, 0, max(0, reveal - edge), h], fill=255)
            # soft edge
            for i in range(edge):
                x = reveal - edge + i
                if 0 <= x < w:
                    md.line([(x, 0), (x, h)], fill=int(255 * i / edge))
        a = img.getchannel("A")
        img.putalpha(Image.composite(a, Image.new("L", (w, h), 0), mask))
        return img

    def _dim_top(self, img: Image.Image) -> Image.Image:
        """Hide the key layer while the script is still appearing."""
        return img

    def _apply_alpha_shift(self, img: Image.Image, alpha: float, dy: int):
        if dy:
            shifted = Image.new("RGBA", img.size, (0, 0, 0, 0))
            shifted.alpha_composite(img, (0, dy))
            img = shifted
        if alpha < 1.0:
            a = img.getchannel("A").point(lambda v: int(v * alpha))
            img.putalpha(a)
        return img


def caption_fonts_for(cfg: dict, fonts: dict[str, Path], script: str, key: str):
    if not script or not key:
        raise ValueError(
            "fonts.script and fonts.key are not set. Render font_match_sheet.png "
            "and choose a pair before rendering (Section 6.2)."
        )
    return {"script_font": script, "key_font": key}
