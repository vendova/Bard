"""Stage 5 — Renderer.

Per shot: trim -> fit frame -> blur-flash settle -> light grade -> overlay the
Pillow caption sequence -> concat -> watermark -> mux processed audio -> fades.

Renderer choice: FFmpeg for picture/audio assembly + Pillow for the lyric
lockup. Rationale: FFmpeg gives frame-accurate trimming and the tested
blur/eq/gblur chain from Section 5.4; Pillow renders complex script fonts and
the wipe/scale animation precisely, avoiding drawtext's limits (Section 10
Stage 5 recommends exactly this hybrid).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from caption_gen import CaptionRenderer, caption_fonts_for
from util import (MissingAssetError, ensure_dirs, licence_for,
                  load_config, load_licences, read_json, require_ffmpeg,
                  require_fonts, resolve, run, write_json)


def _probe_frames(path: str, ffprobe: str) -> int:
    import json
    res = run([ffprobe, "-v", "error", "-select_streams", "v:0",
               "-count_frames", "-show_entries", "stream=nb_read_frames",
               "-of", "json", path])
    try:
        return int(json.loads(res.stdout)["streams"][0]["nb_read_frames"])
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Per-shot picture
# --------------------------------------------------------------------------- #
def build_video_filter(cfg: dict, width: int, height: int, settle: int,
                       first_shot: bool, vertical: bool,
                       face_cx: float, face_cy: float) -> str:
    t = cfg["transition"]
    s = t["strength"]
    wf = width / 1280.0  # blur sigma scaling
    S = max(2, settle)
    d = f"max(0\\,1-n/{S - 1})"
    parts = ["setpts=PTS-STARTPTS"]

    # Fit to frame.
    if vertical:
        parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:(iw-{width})*{min(max(face_cx,0),1):.3f}:"
            f"(ih-{height})*{min(max(face_cy,0.45),1):.3f}"
        )
    else:
        parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )

    if not first_shot and s > 0:
        zoom = t["zoom_amount"] * s
        parts.append(
            f"scale=w='trunc({width}*(1+{zoom}*pow({d},2))/2)*2':"
            f"h='trunc({height}*(1+{zoom}*pow({d},2))/2)*2':eval=frame,"
            f"crop={width}:{height}"
        )
        for band in t["blur_sigma_by_frame"]:
            sig = band["sigma"] * wf * s
            lo, hi = band["frames"][0], band["frames"][1]
            parts.append(f"gblur=sigma={sig:.2f}:enable='between(n,{lo},{hi})'")
        parts.append(
            f"eq=eval=frame:brightness='{t['flash_brightness'] * s}*pow({d},2)':"
            f"saturation='1+{t['saturation_boost'] * s}*{d}'"
        )

    # Light warm grade + sharpen (Section 7).
    g = cfg["grade"]
    warm = g["warm_bias_saturation"]
    parts.append(f"eq=saturation={1 + warm:.3f}:gamma=1.0")
    parts.append(f"colorbalance=rs={t['warm_bias_red'] * 0.5:.3f}:gs=0:bs=-0.01")
    if g["sharpen"]["enabled"]:
        parts.append(f"unsharp=5:5:{g['sharpen']['amount']:.2f}")
    parts.append("format=yuv420p")
    return ",".join(parts)


def render_shot(cfg: dict, ffmpeg: str, shot: dict, idx: int, work: Path,
                width: int, height: int, fps: int, vertical: bool,
                caption_png_dir: Path | None) -> Path:
    settle = cfg["transition"]["settle_frames"]
    if shot["length"] < cfg["transition"]["short_shot_threshold_s"]:
        settle = cfg["transition"]["settle_frames_short"]
    first = idx == 0
    vf = build_video_filter(cfg, width, height, settle, first, vertical,
                            shot.get("face_cx", 0.5), shot.get("face_cy", 0.5))
    n_frames = max(1, int(round(shot["length"] * fps)))
    out = work / f"shot_{idx:03d}.mkv"
    cmd = [ffmpeg, "-y", "-v", "error",
           "-ss", f"{shot['source_in']:.4f}", "-i", shot["source_file"],
           "-frames:v", str(n_frames),
           "-vf", vf, "-r", str(fps),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", str(out)]
    run(cmd)

    if caption_png_dir is not None and shot.get("caption"):
        over = work / f"shot_{idx:03d}_cap.mkv"
        run([ffmpeg, "-y", "-v", "error",
             "-i", str(out),
             "-framerate", str(fps), "-i", str(caption_png_dir / "%05d.png"),
             "-filter_complex",
             "[0:v][1:v]overlay=0:0:format=auto:shortest=1,format=yuv420p[v]",
             "-map", "[v]", "-r", str(fps),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             str(over)])
        return over
    return out


# --------------------------------------------------------------------------- #
# Audio processing (Section 8.3)
# --------------------------------------------------------------------------- #
def process_audio(cfg: dict, ffmpeg: str, music: str, start: float, dur: float,
                  work: Path) -> Path:
    """Trim the window, fade, and normalise with two-pass loudnorm (accurate)."""
    out = work / "audio.m4a"
    fi = cfg["audio_analysis"]["fade_in_ms"] / 1000.0
    fo = cfg["audio_analysis"]["fade_out_ms"] / 1000.0
    target = cfg["loudness"]["target_lufs"]
    tp = cfg["loudness"]["max_true_peak_dbtp"]
    pre = (f"afade=t=in:st=0:d={fi:.3f},"
           f"afade=t=out:st={max(0, dur - fo):.3f}:d={fo:.3f}")
    src = ["-ss", f"{start:.4f}", "-t", f"{dur:.4f}", "-i", music]
    # Pass 1: measure.
    import json, re
    res = run([ffmpeg, "-y", "-v", "info", *src, "-af",
               f"{pre},loudnorm=I={target}:TP={tp}:LRA=11:print_format=json",
               "-f", "null", "-"], check=False)
    m = re.search(r"\{[^{}]*input_i[^{}]*\}", res.stderr, re.S)
    measured = {}
    if m:
        try:
            measured = json.loads(m.group(0))
        except Exception:
            measured = {}
    if measured.get("input_i"):
        ln = (f"loudnorm=I={target}:TP={tp}:LRA=11:"
              f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
              f"measured_LRA={measured['input_lra']}:"
              f"measured_thresh={measured['input_thresh']}:"
              f"offset={measured.get('target_offset', 0)}:linear=true")
        af = f"{pre},{ln}"
    else:  # fall back to single pass
        af = f"{pre},loudnorm=I={target}:TP={tp}:LRA=11"
    # A limiter guarantees the true-peak ceiling holds after AAC (which can push
    # inter-sample peaks up). Set just below the ceiling so the result stays legal.
    af += f",alimiter=limit={10 ** ((tp - 0.2) / 20):.5f}:level=disabled"
    run([ffmpeg, "-y", "-v", "error", *src, "-af", af,
         "-c:a", "aac", "-b:a", f"{cfg['output']['audio_bitrate_k']}k",
         "-ar", str(cfg["output"]["audio_rate"]),
         "-ac", str(cfg["output"]["audio_channels"]), str(out)])
    return out


# --------------------------------------------------------------------------- #
# Watermark (Section 9)
# --------------------------------------------------------------------------- #
def render_watermark(cfg: dict, fonts: dict[str, Path], width: int, height: int,
                     vertical: bool) -> Path:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    wm = cfg["watermark"]
    handle = cfg["project"]["handle"]
    cap_px = wm["cap_height_frac"] * height
    font_path = str(fonts[cfg["fonts"]["watermark_font"]])
    size = max(8, int(cap_px * 1.9))
    font = ImageFont.truetype(font_path, size)
    text = " ".join(handle.upper())  # crude +60 tracking via spacing
    img = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    pad = 20
    layer = Image.new("RGBA", (bbox[2] - bbox[0] + pad * 2,
                               bbox[3] - bbox[1] + pad * 2), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    from caption_gen import _hex
    col = _hex(wm["color"]) + (int(255 * wm["opacity"]),)
    dl.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=col)
    if wm["shadow"]["enabled"]:
        sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ds = ImageDraw.Draw(sh)
        sc = _hex(wm["shadow"]["color"]) + (int(255 * wm["shadow"]["opacity"]),)
        ds.text((pad - bbox[0] + wm["shadow"]["dx"],
                 pad - bbox[1] + wm["shadow"]["dy"]), text, font=font, fill=sc)
        sh = sh.filter(ImageFilter.GaussianBlur(wm["shadow"]["blur"]))
        layer = Image.alpha_composite(sh, layer)
    out = Path(cfg["_root"]) / cfg["paths"]["work_dir"] / "watermark.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    layer.save(out)
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def concat_shots(ffmpeg: str, shots: list[Path], work: Path) -> Path:
    listfile = work / "concat.txt"
    listfile.write_text("".join(f"file '{p}'\n" for p in shots), encoding="utf-8")
    out = work / "picture.mkv"
    run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listfile), "-c", "copy", str(out)])
    return out


def finalize(cfg: dict, ffmpeg: str, ffprobe: str, picture: Path, audio: Path,
             watermark: Path, out_path: Path, width: int, height: int, fps: int,
             vertical: bool, duration: float) -> None:
    wm = cfg["watermark"]
    if vertical:
        y_frac = wm["vertical_y_frac"]
    else:
        y_frac = (wm["bottom_frac"][0] + wm["bottom_frac"][1]) / 2.0
    fade = cfg["output"]["end_fade_frames"]
    fade_start = max(0.0, duration - fade / fps)
    vf = (f"overlay=x=(W-w)/2:y=H*{y_frac:.3f}-h/2:format=auto,"
          f"fade=t=out:st={fade_start:.4f}:d={fade / fps:.4f}")
    run([ffmpeg, "-y", "-v", "error", "-i", str(picture), "-i", str(watermark),
         "-i", str(audio),
         "-filter_complex", f"[0:v][1:v]{vf}[v]",
         "-map", "[v]", "-map", "2:a",
         "-t", f"{duration:.4f}",
         "-c:v", cfg["output"]["codec"], "-profile:v", cfg["output"]["profile"],
         "-b:v", f"{cfg['output']['bitrate_mbps']}M",
         "-pix_fmt", cfg["output"]["pix_fmt"],
         "-color_primaries", cfg["output"]["color_primaries"],
         "-color_trc", cfg["output"]["color_trc"],
         "-colorspace", cfg["output"]["colorspace"],
         "-c:a", cfg["output"]["audio_codec"],
         "-b:a", f"{cfg['output']['audio_bitrate_k']}k",
         str(out_path)])


# --------------------------------------------------------------------------- #
# Top-level render of a plan
# --------------------------------------------------------------------------- #
def render_plan(cfg: dict, plan: dict, music: str, caption_file: str | None,
                out_name: str | None = None) -> Path:
    ensure_dirs(cfg)
    ffmpeg, ffprobe = require_ffmpeg()
    fonts = require_fonts(cfg)

    # Licence gate (Section 1.1) — music + footage + captions.
    licences = load_licences(cfg)
    for asset in [music] + [s["source_file"] for s in plan["shots"]]:
        licence_for(cfg, licences, asset)
    if caption_file:
        licence_for(cfg, licences, caption_file)

    script = cfg["fonts"]["script"]
    key = cfg["fonts"]["key"]
    if not script or not key:
        raise MissingAssetError(
            "fonts.script/fonts.key unset. Run `python src/font_match.py` and "
            "set the pair in config.yaml before rendering (Section 6.2)."
        )
    font_sel = caption_fonts_for(cfg, fonts, script, key)

    out_cfg = cfg["output"]
    vertical = bool(out_cfg["vertical"])
    width, height = ((1080, 1920) if vertical else (out_cfg["width"], out_cfg["height"]))
    fps = out_cfg["fps"]

    work = resolve(cfg, "work_dir")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    capr = CaptionRenderer(cfg, fonts, width, height, vertical=vertical)
    shot_files: list[Path] = []
    for i, shot in enumerate(plan["shots"]):
        cap_dir = None
        if shot.get("caption"):
            cap = {**shot["caption"], **font_sel}
            n_frames = max(1, int(round(shot["length"] * fps)))
            frames = capr.frames_for_shot(cap, n_frames, cut_is_first=(i == 0))
            cap_dir = work / f"cap_{i:03d}"
            cap_dir.mkdir(exist_ok=True)
            for j, fr in enumerate(frames):
                fr.save(cap_dir / f"{j:05d}.png")
        sf = render_shot(cfg, ffmpeg, shot, i, work, width, height, fps,
                         vertical, cap_dir)
        shot_files.append(sf)
        print(f"  shot {i + 1}/{len(plan['shots'])} -> {sf.name}")

    picture = concat_shots(ffmpeg, shot_files, work)
    start = plan["shots"][0]["timeline_in"]
    dur = plan["shots"][-1]["timeline_out"] - start
    audio = process_audio(cfg, ffmpeg, music, start, dur, work)
    wm = render_watermark(cfg, fonts, width, height, vertical)

    renders = resolve(cfg, "renders_dir")
    renders.mkdir(parents=True, exist_ok=True)
    out_path = renders / (out_name or f"{Path(music).stem}_{plan['layout']}_{plan['seed']}.mp4")
    finalize(cfg, ffmpeg, ffprobe, picture, audio, wm, out_path, width, height,
             fps, vertical, dur)
    print(f"✅ rendered -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Stage 5: render a plan")
    ap.add_argument("--config", default=None)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--music", required=True)
    ap.add_argument("--captions", default=None)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    plan = read_json(args.plan)
    render_plan(cfg, plan, args.music, args.captions, args.out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
