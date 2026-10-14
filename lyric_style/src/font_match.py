"""Font contact sheet — 4 script candidates x 4 key candidates (Section 6.2).

Renders every script/key pair over one still frame from the footage pool and
saves font_match_sheet.png. The user picks the pair; it is then stored in
config.yaml as fonts.script / fonts.key. Batch rendering must not start until a
pair is set.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from caption_gen import CaptionRenderer
from util import load_config, require_fonts, resolve


def _grab_still(footage_dir: Path, ffmpeg: str, width: int, height: int) -> Image.Image:
    import subprocess
    clips = [p for p in sorted(footage_dir.glob("*"))
             if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}]
    if not clips:
        # Plain backdrop if no footage yet.
        return Image.new("RGB", (width, height), (40, 30, 45))
    clip = clips[0]
    out = "/tmp/_fontsheet_still.png"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", "1.0", "-i", str(clip),
                    "-frames:v", "1", "-vf", f"scale={width}:{height}"
                    ":force_original_aspect_ratio=increase,"
                    f"crop={width}:{height}", out], check=True)
    return Image.open(out).convert("RGB")


def build_sheet(cfg: dict, out_path: str | None = None) -> Path:
    from util import require_ffmpeg
    ffmpeg, _ = require_ffmpeg()
    fonts = require_fonts(cfg)
    W, H = 1920, 1080
    base = _grab_still(resolve(cfg, "footage_dir"), ffmpeg, W, H)
    script_c = cfg["fonts"]["script_candidates"]
    key_c = cfg["fonts"]["key_candidates"]
    cell_w, cell_h = W // len(key_c), H // len(script_c)
    sheet = Image.new("RGB", (W, H), (0, 0, 0))
    rng = random.Random(0)
    sample = {"script": "Forever", "key": "YOURS", "anchor": "right"}
    for r, script in enumerate(script_c):
        for c, key in enumerate(key_c):
            tile = base.resize((cell_w, cell_h))
            cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
            cell.paste(tile, (0, 0))
            sub = dict(cfg)
            # Render the lockup at tile scale by temporarily shrinking the frame.
            capr = CaptionRenderer(cfg, fonts, cell_w, cell_h, vertical=False)
            lock = capr.compose_static({**sample, **{"script_font": script,
                                                     "key_font": key}})
            cell.alpha_composite(lock, (0, 0))
            cell = cell.convert("RGB")
            d = ImageDraw.Draw(cell)
            f = ImageFont.truetype(str(fonts[cfg["fonts"]["watermark_font"]]), 22)
            d.rectangle([0, 0, cell_w, 30], fill=(0, 0, 0))
            d.text((8, 4), f"{script.split('-')[0]}  x  {key.split('-')[0]}",
                   font=f, fill=(255, 255, 255))
            sheet.paste(cell, (c * cell_w, r * cell_h))
    out = Path(out_path) if out_path else (Path(cfg["_root"]) / "font_match_sheet.png")
    sheet.save(out)
    print(f"✅ font_match_sheet.png -> {out}")
    print("   Choose a pair and set fonts.script / fonts.key in config.yaml.")
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Render the font contact sheet")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    build_sheet(cfg, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
