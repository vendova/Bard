#!/usr/bin/env python3
"""
Video Effects Engine for BeatSync-Engine.

Provides per-clip visual effects (gradient overlays, vignette, zoom punch,
shake, flash, slow motion, color grading, glitch) and inter-clip transition
effects (crossfade, wipe, slide, zoom, dissolve) applied during rendering.

Per-clip effects are injected into the FFmpeg -vf filter chain during segment
extraction so the existing frame-accurate concat pipeline stays intact.
Transitions use FFmpeg's xfade filter in a dedicated re-encode pass.
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import ffmpeg_processing as fp


# ---------------------------------------------------------------------------
# Transition definitions
# ---------------------------------------------------------------------------

TRANSITION_TYPES = {
    'none':        'No transition (hard cut)',
    'fade':        'Crossfade dissolve',
    'fadeblack':   'Fade through black',
    'fadewhite':   'Fade through white',
    'wipeleft':    'Wipe left',
    'wiperight':   'Wipe right',
    'wipeup':      'Wipe up',
    'wipedown':    'Wipe down',
    'slideleft':   'Slide left',
    'slideright':  'Slide right',
    'slideup':     'Slide up',
    'slidedown':   'Slide down',
    'circleopen':  'Circle open',
    'circleclose': 'Circle close',
    'smoothleft':  'Smooth zoom left',
    'smoothright': 'Smooth zoom right',
    'smoothup':    'Smooth zoom up',
    'smoothdown':  'Smooth zoom down',
    'dissolve':    'Pixel dissolve',
    'radial':      'Radial wipe',
    'zoomin':      'Zoom in transition',
}

# Map our names to FFmpeg xfade transition names
_XFADE_MAP = {
    'fade': 'fade', 'fadeblack': 'fadeblack', 'fadewhite': 'fadewhite',
    'wipeleft': 'wipeleft', 'wiperight': 'wiperight',
    'wipeup': 'wipeup', 'wipedown': 'wipedown',
    'slideleft': 'slideleft', 'slideright': 'slideright',
    'slideup': 'slideup', 'slidedown': 'slidedown',
    'circleopen': 'circleopen', 'circleclose': 'circleclose',
    'smoothleft': 'smoothleft', 'smoothright': 'smoothright',
    'smoothup': 'smoothup', 'smoothdown': 'smoothdown',
    'dissolve': 'dissolve', 'radial': 'radial',
    'zoomin': 'zoomin',
}


# ---------------------------------------------------------------------------
# Effect configuration
# ---------------------------------------------------------------------------

@dataclass
class EffectsConfig:
    """User-facing effect settings, threaded through the render pipeline."""

    # Master toggle
    enabled: bool = False

    # --- Per-clip visual effects ---
    gradient_overlay: bool = False
    gradient_colors: str = 'auto'          # 'auto' or hex like '0xff00aa-0x00aaff'
    vignette: bool = False
    vignette_strength: float = 0.4         # 0-1
    zoom_punch: bool = False               # subtle zoom in/out per clip
    zoom_punch_strength: float = 1.06      # max scale factor
    shake: bool = False                    # camera shake on beats
    shake_strength: float = 8.0            # pixels
    flash_on_beat: bool = False            # white flash at clip start
    flash_intensity: float = 0.6           # 0-1
    color_boost: bool = False              # saturation/contrast boost
    color_boost_amount: float = 1.3        # multiplier
    glitch: bool = False                   # digital glitch effect
    glitch_strength: float = 0.3           # 0-1
    mirror: bool = False                   # random horizontal flip

    # --- Slow motion ---
    slow_motion: bool = False
    slow_motion_factor: float = 2.0        # 2x = half speed
    slow_motion_probability: float = 0.15  # fraction of clips affected

    # --- Transitions ---
    transition_type: str = 'none'
    transition_duration: float = 0.35      # seconds


def parse_effects_from_ui(
    enabled: bool,
    gradient_overlay: bool,
    vignette: bool,
    zoom_punch: bool,
    shake: bool,
    flash_on_beat: bool,
    color_boost: bool,
    glitch: bool,
    mirror: bool,
    slow_motion: bool,
    slow_motion_factor: float,
    transition_type: str,
    transition_duration: float,
    vignette_strength: float = 0.4,
    zoom_punch_strength: float = 1.06,
    shake_strength: float = 8.0,
    flash_intensity: float = 0.6,
    color_boost_amount: float = 1.3,
    glitch_strength: float = 0.3,
    slow_motion_probability: float = 0.15,
) -> EffectsConfig:
    """Build an EffectsConfig from raw Gradio UI values."""
    return EffectsConfig(
        enabled=bool(enabled),
        gradient_overlay=bool(gradient_overlay),
        vignette=bool(vignette),
        vignette_strength=float(vignette_strength),
        zoom_punch=bool(zoom_punch),
        zoom_punch_strength=float(zoom_punch_strength),
        shake=bool(shake),
        shake_strength=float(shake_strength),
        flash_on_beat=bool(flash_on_beat),
        flash_intensity=float(flash_intensity),
        color_boost=bool(color_boost),
        color_boost_amount=float(color_boost_amount),
        glitch=bool(glitch),
        glitch_strength=float(glitch_strength),
        mirror=bool(mirror),
        slow_motion=bool(slow_motion),
        slow_motion_factor=float(slow_motion_factor),
        slow_motion_probability=float(slow_motion_probability),
        transition_type=str(transition_type) if transition_type else 'none',
        transition_duration=float(transition_duration),
    )


# ---------------------------------------------------------------------------
# Per-clip filter generation
# ---------------------------------------------------------------------------

# A pool of vibrant gradient color pairs for auto mode
_GRADIENT_PALETTE = [
    ('0x00f5ff', '0xff00e5'),   # cyan → magenta
    ('0xff4500', '0xffd700'),   # orange → gold
    ('0x4b0082', '0xff1493'),   # indigo → pink
    ('0x00ff7f', '0x00bfff'),   # spring green → deep sky
    ('0xff1493', '0x4b0082'),   # pink → indigo
    ('0xffd700', '0xff4500'),   # gold → orange
    ('0x9370db', '0x00ced1'),   # medium purple → dark turquoise
    ('0xdc143c', '0xff8c00'),   # crimson → dark orange
]


def _pick_gradient_colors(clip_index: int) -> Tuple[str, str]:
    return _GRADIENT_PALETTE[clip_index % len(_GRADIENT_PALETTE)]


def build_clip_effect_filters(
    cfg: EffectsConfig,
    clip_index: int,
    clip_duration: float,
    fps: float,
    target_size: Tuple[int, int],
) -> Tuple[List[str], Optional[float]]:
    """
    Build extra FFmpeg -vf filter strings for a single clip.

    Returns (filter_list, speed_factor) where speed_factor is None for normal
    speed or a float > 1.0 when slow motion is applied (caller must extract
    speed_factor * duration of source content so the slowed clip fills the
    target timeline slot).
    """
    if not cfg.enabled:
        return [], None

    filters: List[str] = []
    speed_factor: Optional[float] = None
    rng = random.Random(clip_index * 7919 + 42)
    w, h = target_size if target_size else (1920, 1080)

    # --- Slow motion ---
    # The actual setpts filter is applied by extract_clip_segment_ffmpeg
    # via the speed_factor return value; here we only signal that slow-mo
    # should be used for this clip.
    if cfg.slow_motion and rng.random() < cfg.slow_motion_probability:
        speed_factor = cfg.slow_motion_factor

    # --- Mirror (random horizontal flip) ---
    if cfg.mirror and rng.random() < 0.3:
        filters.append('hflip')

    # --- Color boost (saturation + contrast) ---
    if cfg.color_boost:
        amt = cfg.color_boost_amount
        sat = amt
        contrast = 1.0 + (amt - 1.0) * 0.5
        filters.append(f"eq=saturation={sat:.2f}:contrast={contrast:.2f}")

    # --- Gradient overlay ---
    if cfg.gradient_overlay:
        if cfg.gradient_colors == 'auto' or '-' not in cfg.gradient_colors:
            c1, c2 = _pick_gradient_colors(clip_index)
        else:
            parts = cfg.gradient_colors.split('-')
            c1, c2 = parts[0].strip(), parts[1].strip()
        # Apply a semi-transparent color tint that shifts over time using
        # colorchannelmixer for a vibrant gradient wash.
        r1 = int(c1[2:4], 16) / 255.0
        g1 = int(c1[4:6], 16) / 255.0
        b1 = int(c1[6:8], 16) / 255.0
        mix = 0.15  # subtle 15% blend
        filters.append(
            f"colorchannelmixer=rr={1-mix*(1-r1)}:gg={1-mix*(1-g1)}:bb={1-mix*(1-b1)}"
        )

    # --- Vignette ---
    if cfg.vignette:
        # PI/2 normalised vignette via geq
        cx, cy = w / 2, h / 2
        max_d = (cx ** 2 + cy ** 2) ** 0.5
        strength = cfg.vignette_strength
        filters.append(
            f"geq=lum='p(X,Y)*(1-{strength}*((sqrt((X-{cx})^2+(Y-{cy})^2)/{max_d:.1f})^2))'"
        )

    # --- Zoom punch ---
    if cfg.zoom_punch:
        z = cfg.zoom_punch_strength
        # Zoom in then back out using a sinusoidal zoompan expression.
        total_frames = max(1, int(round(clip_duration * fps)))
        # Use zoompan with a d= duration in frames, oscillating zoom.
        # Simpler: a single subtle zoom-in across the clip.
        filters.append(
            f"zoompan=z='min(zoom+0.0015,{z})':d={total_frames}"
            f":s={w}x{h}:fps={fps}"
        )

    # --- Shake ---
    if cfg.shake:
        s = int(cfg.shake_strength)
        total_frames = max(1, int(round(clip_duration * fps)))
        # Crop a slightly smaller region with sinusoidal offset, then pad back.
        # FFmpeg crop x/y expressions use 'n' (frame index) for oscillation.
        period_x = max(1, total_frames // 4)
        period_y = max(1, total_frames // 5)
        filters.append(
            f"crop=iw-{s*2}:ih-{s*2}:"
            f"'(iw-{s*2})/2+{s}*sin(n/{period_x})*0.5':"
            f"'(ih-{s*2})/2+{s}*cos(n/{period_y})*0.5',"
            f"pad={w}:{h}:{s}:{s}:black"
        )

    # --- Glitch ---
    if cfg.glitch:
        gs = cfg.glitch_strength
        # Digital glitch via periodic hue rotation + temporal pixel lag.
        # Uses freezeframes + hue modulation for a choppy, glitchy look.
        glitch_intensity = int(gs * 30)  # frames to hold
        filters.append(
            f"hue=h='sin(n)*{gs * 180:.0f}':s=1,"
            f"mpdecimate=hi=64*48:lo=64*48:frac=0.33"
        )

    # --- Flash on beat ---
    if cfg.flash_on_beat:
        fi = cfg.flash_intensity
        flash_frames = max(2, int(round(fps * 0.08)))  # ~80ms flash
        total_frames = max(1, int(round(clip_duration * fps)))
        # Flash white at the start of the clip, fading out over flash_frames.
        # Use enable expression on a white fade.
        enable_end = min(flash_frames, total_frames) / fps
        filters.append(
            f"drawbox=t=fill:c=white@{fi}:"
            f"x=0:y=0:w={w}:h={h}:"
            f"enable='between(t,0,{enable_end:.3f})'"
        )

    return filters, speed_factor


# ---------------------------------------------------------------------------
# Transition application (xfade-based concatenation)
# ---------------------------------------------------------------------------

def needs_transitions(cfg: EffectsConfig) -> bool:
    return cfg.enabled and cfg.transition_type != 'none' and cfg.transition_type in _XFADE_MAP


def apply_transitions_ffmpeg(
    clip_files: List[str],
    output_file: str,
    audio_file: Optional[str],
    start_time: float,
    end_time: Optional[float],
    cfg: EffectsConfig,
    fps: float,
    use_nvenc: bool = False,
    gpu_encoder: str = 'h264_nvenc',
    temp_dir: Optional[str] = None,
) -> str:
    """
    Concatenate clips with xfade transitions, then mux audio.

    Each pair of adjacent clips is joined with an xfade of the configured type
    and duration.  The resulting timeline is shorter by (N-1)*transition_dur;
    audio is trimmed to match so sync is preserved at the cost of a small
    overall duration reduction.
    """
    if len(clip_files) < 2:
        # Single clip — nothing to transition; fall back to simple mux.
        return fp.concatenate_videos_ffmpeg(
            clip_files, output_file, audio_file, start_time, end_time,
            use_nvenc, gpu_encoder, fps, temp_dir,
        )

    xfade_name = _XFADE_MAP[cfg.transition_type]
    t_dur = cfg.transition_duration
    os.makedirs(temp_dir or os.path.dirname(output_file) or '.', exist_ok=True)

    # Probe durations of each clip.
    durations = [fp.get_video_duration(c) for c in clip_files]

    # Build xfade chain.  For clips c0..cN with durations d0..dN:
    #   [0][1] xfade=transition=X:duration=T:offset=d0-T  → [v01]  (len d0+d1-T)
    #   [v01][2] xfade=...:offset=(d0+d1-T)-T              → [v012] (len d0+d1+d2-2T)
    filter_parts: List[str] = []
    inputs: List[str] = []

    for i, clip in enumerate(clip_files):
        inputs.extend(['-i', clip])

    # First xfade
    cumulative = durations[0]
    prev_label = '[0:v]'
    for i in range(1, len(clip_files)):
        offset = cumulative - t_dur
        if offset < 0:
            offset = 0
        out_label = f'[v{i}]' if i < len(clip_files) - 1 else '[vout]'
        filter_parts.append(
            f"{prev_label}[{i}:v]xfade=transition={xfade_name}:"
            f"duration={t_dur}:offset={offset:.4f}{out_label}"
        )
        cumulative = cumulative + durations[i] - t_dur
        prev_label = out_label

    filter_complex = ';'.join(filter_parts)

    # Build command: ALL inputs first, then ALL output options.
    cmd = [fp.FFMPEG_PATH, '-nostdin', '-hide_banner']
    cmd.extend(inputs)

    # Audio as additional input (must come before output options).
    audio_input_idx = len(clip_files)
    audio_filters: List[str] = []
    has_audio = False
    if audio_file:
        cmd.extend(['-i', audio_file])
        has_audio = True
        trim_end = cumulative if end_time and end_time > start_time else None
        if start_time > 0 or trim_end:
            parts = []
            if start_time > 0:
                parts.append(f'atrim=start={start_time}')
            if trim_end:
                a_dur = trim_end - start_time
                parts.append(f'atrim=duration={a_dur}')
            parts.append('asetpts=PTS-STARTPTS')
            audio_filters.append(','.join(parts))

    # Output options: filter_complex, maps, codecs.
    cmd.extend(['-filter_complex', filter_complex, '-map', '[vout]'])
    if has_audio:
        cmd.extend(['-map', f'{audio_input_idx}:a'])
        if audio_filters:
            cmd.extend(['-af', ';'.join(audio_filters)])
        cmd.extend(['-c:a', 'aac', '-b:a', '192k', '-shortest'])
    else:
        cmd.extend(['-an'])

    # Video encoding
    if use_nvenc:
        cmd.extend(fp.get_nvenc_clip_quality_args(gpu_encoder, include_pix_fmt=True))
    else:
        cmd.extend(fp.get_clip_h264_quality_args(include_pix_fmt=True))

    cmd.extend([
        '-r', str(fps),
        '-fps_mode', 'cfr',
        '-movflags', '+faststart',
        '-y', output_file,
    ])

    print(f"   🎭 Applying {xfade_name} transitions ({len(clip_files)-1} joins, "
          f"{t_dur}s each)...")
    result = fp._run_media_command(cmd, timeout=1800)

    if result.returncode != 0:
        err = fp._short_ffmpeg_error(result.stderr)
        print(f"   ⚠️  Transition xfade failed ({err}); falling back to hard-cut concat.")
        return fp.concatenate_videos_ffmpeg(
            clip_files, output_file, audio_file, start_time, end_time,
            use_nvenc, gpu_encoder, fps, temp_dir,
        )

    final_dur = fp.get_video_duration(output_file)
    print(f"   ✓ Transitions applied. Output duration: {final_dur:.2f}s "
          f"(was {sum(durations):.2f}s, saved {(sum(durations) - final_dur):.2f}s)")
    return output_file


def smart_transitions_ffmpeg(
    clip_files: List[str],
    output_file: str,
    audio_file: Optional[str],
    start_time: float,
    end_time: Optional[float],
    cfg: EffectsConfig,
    transition_types: List[str],
    transition_durations: Optional[List[float]] = None,
    fps: float = 30.0,
    use_nvenc: bool = False,
    gpu_encoder: str = 'h264_nvenc',
    temp_dir: Optional[str] = None,
) -> str:
    """Concatenate clips with per-cut xfade transitions (Smart Sync mode).

    Like ``apply_transitions_ffmpeg`` but each cut can use a different
    transition type and duration, chosen by the smart sync engine.

    Args:
        transition_types: one transition name per cut (len = len(clip_files) - 1).
        transition_durations: optional per-cut duration; falls back to cfg.
    """
    n = len(clip_files)
    if n < 2:
        return fp.concatenate_videos_ffmpeg(
            clip_files, output_file, audio_file, start_time, end_time,
            use_nvenc, gpu_encoder, fps, temp_dir,
        )

    # Validate transition list length.
    if len(transition_types) < n - 1:
        transition_types = transition_types + ["fade"] * (n - 1 - len(transition_types))
    transition_types = [t if t in _XFADE_MAP else "fade" for t in transition_types[:n - 1]]

    if transition_durations is None:
        transition_durations = [cfg.transition_duration] * (n - 1)
    else:
        transition_durations = (transition_durations[:n - 1]
                                + [cfg.transition_duration] * max(0, n - 1 - len(transition_durations)))

    os.makedirs(temp_dir or os.path.dirname(output_file) or '.', exist_ok=True)

    durations = [fp.get_video_duration(c) for c in clip_files]

    filter_parts: List[str] = []
    inputs: List[str] = []
    for clip in clip_files:
        inputs.extend(['-i', clip])

    cumulative = durations[0]
    prev_label = '[0:v]'
    for i in range(1, n):
        t_name = _XFADE_MAP[transition_types[i - 1]]
        t_dur = transition_durations[i - 1]
        offset = max(0.0, cumulative - t_dur)
        out_label = f'[v{i}]' if i < n - 1 else '[vout]'
        filter_parts.append(
            f"{prev_label}[{i}:v]xfade=transition={t_name}:"
            f"duration={t_dur:.4f}:offset={offset:.4f}{out_label}"
        )
        cumulative = cumulative + durations[i] - t_dur
        prev_label = out_label

    filter_complex = ';'.join(filter_parts)

    cmd = [fp.FFMPEG_PATH, '-nostdin', '-hide_banner']
    cmd.extend(inputs)

    audio_input_idx = n
    audio_filters: List[str] = []
    has_audio = False
    if audio_file:
        cmd.extend(['-i', audio_file])
        has_audio = True
        trim_end = cumulative if end_time and end_time > start_time else None
        if start_time > 0 or trim_end:
            parts = []
            if start_time > 0:
                parts.append(f'atrim=start={start_time}')
            if trim_end:
                a_dur = trim_end - start_time
                parts.append(f'atrim=duration={a_dur}')
            parts.append('asetpts=PTS-STARTPTS')
            audio_filters.append(','.join(parts))

    cmd.extend(['-filter_complex', filter_complex, '-map', '[vout]'])
    if has_audio:
        cmd.extend(['-map', f'{audio_input_idx}:a'])
        if audio_filters:
            cmd.extend(['-af', ';'.join(audio_filters)])
        cmd.extend(['-c:a', 'aac', '-b:a', '192k', '-shortest'])
    else:
        cmd.extend(['-an'])

    if use_nvenc:
        cmd.extend(fp.get_nvenc_clip_quality_args(gpu_encoder, include_pix_fmt=True))
    else:
        cmd.extend(fp.get_clip_h264_quality_args(include_pix_fmt=True))

    cmd.extend([
        '-r', str(fps),
        '-fps_mode', 'cfr',
        '-movflags', '+faststart',
        '-y', output_file,
    ])

    # Summarise the transition mix.
    from collections import Counter
    mix = Counter(transition_types)
    mix_str = ", ".join(f"{t}×{c}" for t, c in mix.items())
    print(f"   🧠 Smart transitions: {n - 1} cuts [{mix_str}]")

    result = fp._run_media_command(cmd, timeout=1800)

    if result.returncode != 0:
        err = fp._short_ffmpeg_error(result.stderr)
        print(f"   ⚠️  Smart transition xfade failed ({err}); falling back to hard-cut concat.")
        return fp.concatenate_videos_ffmpeg(
            clip_files, output_file, audio_file, start_time, end_time,
            use_nvenc, gpu_encoder, fps, temp_dir,
        )

    final_dur = fp.get_video_duration(output_file)
    print(f"   ✓ Smart transitions applied. Output duration: {final_dur:.2f}s "
          f"(was {sum(durations):.2f}s, saved {(sum(durations) - final_dur):.2f}s)")
    return output_file


# ---------------------------------------------------------------------------
# Fade in/out per clip (lightweight transition that preserves concat pipeline)
# ---------------------------------------------------------------------------

def build_fade_filters(clip_index: int, clip_duration: float, fps: float,
                       fade_duration: float = 0.15) -> List[str]:
    """Build fade-in / fade-out filters for a clip (dissolve-to-black style)."""
    if clip_duration <= fade_duration * 2:
        return []
    fade_frames = max(1, int(round(fade_duration * fps)))
    return [
        f"fade=t=in:st=0:d={fade_duration:.3f}",
        f"fade=t=out:st={clip_duration - fade_duration:.3f}:d={fade_duration:.3f}",
    ]
