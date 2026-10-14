"""
Lyrics Engine — time-synced lyric extraction, transliteration, and ASS
subtitle generation for overlaying lyrics on the music video.

Workflow:
1. Parse lyrics from plain text, LRC, or timestamped format
2. Auto-time lyrics to beat times / sections when no timestamps provided
3. Transliterate non-English lyrics to Latin script
4. Generate ASS subtitle file with the selected font combo preset
5. Burn subtitles onto the video via FFmpeg's subtitles/ass filter
"""

from __future__ import annotations

import os
import re
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from transliteration import transliterate, needs_transliteration, detect_language
from font_presets import FontCombo, get_combo_by_name, get_combo, FONTS_DIR
from transition_presets import TransitionPreset, get_preset_by_name, get_preset


# ---------------------------------------------------------------------------
# Lyric line data structure
# ---------------------------------------------------------------------------

@dataclass
class LyricLine:
    text: str = ""
    transliterated: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    is_primary: bool = True   # True = main line, False = secondary/accent

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "transliterated": self.transliterated,
            "start": round(self.start_time, 3),
            "end": round(self.end_time, 3),
        }


# ---------------------------------------------------------------------------
# Lyrics parsing
# ---------------------------------------------------------------------------

def parse_lrc(content: str) -> List[LyricLine]:
    """Parse LRC format lyrics with timestamps.

    Supports standard [mm:ss.xx] and [mm:ss] timestamp tags.
    """
    lines = []
    time_pattern = re.compile(r'\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]')
    pending_times: List[float] = []

    for raw_line in content.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Extract all timestamps from the line (LRC allows multiple per line)
        matches = list(time_pattern.finditer(raw_line))
        if not matches:
            # Untimed line — skip or add as untimed
            continue

        # Remove metadata tags like [ar:], [ti:], etc.
        if re.match(r'^\[(ar|ti|al|by|offset|length|re|ve):', raw_line, re.IGNORECASE):
            continue

        text_part = time_pattern.sub('', raw_line).strip()
        for m in matches:
            mins = int(m.group(1))
            secs = int(m.group(2))
            frac = m.group(3)
            if frac:
                frac_val = int(frac) / (10 ** len(frac))
            else:
                frac_val = 0.0
            t = mins * 60 + secs + frac_val
            lines.append(LyricLine(text=text_part, start_time=t))

    # Sort by time and set end times
    lines.sort(key=lambda l: l.start_time)
    for i in range(len(lines) - 1):
        lines[i].end_time = lines[i + 1].start_time
    if lines:
        lines[-1].end_time = lines[-1].start_time + 5.0  # default 5s for last line

    return lines


def parse_plain_lyrics(content: str, audio_duration: float = 180.0,
                       beat_times: Optional[List[float]] = None) -> List[LyricLine]:
    """Parse plain text lyrics and auto-distribute timing across the song.

    If beat_times are provided, aligns lines to beat boundaries.
    Otherwise, distributes evenly across audio_duration.
    """
    raw_lines = [l.strip() for l in content.split('\n') if l.strip()]
    if not raw_lines:
        return []

    # Filter out empty lines but track verse breaks (double newlines)
    lines: List[LyricLine] = []
    for raw in raw_lines:
        # Skip section markers like [Verse], [Chorus]
        if re.match(r'^\[.*\]$', raw):
            continue
        lines.append(LyricLine(text=raw))

    if not lines:
        return []

    n = len(lines)
    line_dur = max(4.0, audio_duration / max(1, n))

    if beat_times is not None and len(beat_times) > n:
        # Align to beats — pick evenly spaced beats
        step = len(beat_times) / n
        for i, line in enumerate(lines):
            beat_idx = int(i * step)
            line.start_time = beat_times[beat_idx]
            next_idx = int((i + 1) * step)
            if next_idx < len(beat_times):
                line.end_time = beat_times[next_idx]
            else:
                line.end_time = line.start_time + (audio_duration - line.start_time) / max(1, n - i)
    else:
        # Even distribution across audio duration
        # Leave 2s intro and 2s outro padding
        usable = max(10.0, audio_duration - 4.0)
        line_dur = usable / n
        for i, line in enumerate(lines):
            line.start_time = 2.0 + i * line_dur
            line.end_time = line.start_time + line_dur

    # Set last line end time
    if lines:
        lines[-1].end_time = min(audio_duration, lines[-1].start_time + max(4.0, line_dur))

    return lines


def parse_lyrics(content: str, audio_duration: float = 180.0,
                 beat_times: Optional[List[float]] = None) -> List[LyricLine]:
    """Auto-detect format and parse lyrics.

    Supports LRC format ([mm:ss.xx] timestamps) and plain text.
    """
    if not content or not content.strip():
        return []

    # Check if it looks like LRC (has timestamp tags)
    if re.search(r'\[\d{1,2}:\d{2}', content):
        lrc_lines = parse_lrc(content)
        if lrc_lines:
            return lrc_lines

    return parse_plain_lyrics(content, audio_duration, beat_times)


# ---------------------------------------------------------------------------
# Transliteration integration
# ---------------------------------------------------------------------------

def apply_transliteration(lines: List[LyricLine]) -> List[LyricLine]:
    """Transliterate all lyric lines to Latin script if needed."""
    for line in lines:
        if needs_transliteration(line.text):
            line.transliterated = transliterate(line.text)
        else:
            line.transliterated = line.text
    return lines


# ---------------------------------------------------------------------------
# LRC output (timestamps the user can read / reuse)
# ---------------------------------------------------------------------------

def lrc_timestamp(t: float) -> str:
    """Format seconds as an LRC tag: [mm:ss.xx]."""
    t = max(0.0, float(t))
    mins = int(t // 60)
    secs = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:          # rounding can push .xx to 1.00
        cs = 0
        secs += 1
        if secs == 60:
            secs = 0
            mins += 1
    return f"[{mins:02d}:{secs:02d}.{cs:02d}]"


def format_lrc(lines: List[LyricLine], title: str = "",
               include_transliteration: bool = False) -> str:
    """Render lyric lines as an LRC document with ``[mm:ss.xx]`` timestamps.

    ``include_transliteration`` appends the romanized text in parentheses so a
    viewer can read a non-Latin song's pronunciation while the original script
    stays the primary line.
    """
    header = ["[re:BeatSync Engine]"]
    if title:
        header.insert(0, f"[ti:{title}]")
    body = []
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if include_transliteration and line.transliterated and line.transliterated != line.text:
            text = f"{text} ({line.transliterated.strip()})"
        body.append(f"{lrc_timestamp(line.start_time)}{text}")
    return "\n".join(header + body) + "\n"


def write_lrc_file(lines: List[LyricLine], output_path: str, title: str = "",
                   include_transliteration: bool = False) -> str:
    """Write lyric lines to an LRC file and return its path."""
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(format_lrc(lines, title=title,
                            include_transliteration=include_transliteration))
    return output_path


def lines_from_segments(segments: List[Dict]) -> List[LyricLine]:
    """Build LyricLine objects from ``{start, end, text}`` ASR segments."""
    lines: List[LyricLine] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 3.0))
        lines.append(LyricLine(text=text, start_time=start, end_time=max(end, start + 0.2)))
    return lines


def build_lyrics_artifact(
    lyrics_text: str,
    audio_duration: float,
    beat_times: Optional[List[float]] = None,
    audio_file: Optional[str] = None,
    output_dir: str = "/tmp",
    title: str = "",
    language: Optional[str] = None,
    allow_asr: bool = True,
) -> Dict[str, object]:
    """Detect the language, produce timed lyric lines, and write an LRC file.

    When ``lyrics_text`` is empty and ``allow_asr`` is set, an installed
    speech-to-text backend (see :mod:`lyrics_asr`) is used to transcribe
    ``audio_file`` into timed lines. Otherwise the pasted lyrics are parsed
    (LRC tags honoured, plain text auto-distributed across the song).

    Returns a dict with ``lines``, ``lrc_text``, ``lrc_path``, ``language``,
    ``script``, ``indic``, ``source`` ("lyrics" | "asr" | "none") and ``note``.
    """
    import lyrics_asr

    lines: List[LyricLine] = []
    source = "none"
    note = ""

    if lyrics_text and lyrics_text.strip():
        lines = parse_lyrics(lyrics_text, audio_duration, beat_times)
        source = "lyrics" if lines else "none"
    elif allow_asr and audio_file and lyrics_asr.is_available():
        try:
            segments = lyrics_asr.transcribe(audio_file, language=language)
            lines = lines_from_segments(segments)
            source = "asr" if lines else "none"
            note = f"Transcribed with {lyrics_asr.available_backend()}"
        except Exception as exc:                     # never fail the render
            note = f"Auto-transcription failed: {exc}"
    elif allow_asr and audio_file:
        note = ("No speech-to-text backend installed — paste lyrics manually "
                "(pip install -r requirements-lyrics.txt).")

    if lines:
        lines = apply_transliteration(lines)

    detected = detect_language(lyrics_text) if (lyrics_text and lyrics_text.strip()) \
        else detect_language(" ".join(l.text for l in lines))

    lrc_text = format_lrc(lines, title=title) if lines else ""
    lrc_path = ""
    if lrc_text:
        lrc_path = os.path.join(output_dir, f"lyrics_{int(time.time())}.lrc")
        with open(lrc_path, "w", encoding="utf-8") as fh:
            fh.write(lrc_text)

    return {
        "lines": lines,
        "lrc_text": lrc_text,
        "lrc_path": lrc_path,
        "language": detected["language"],
        "script": detected["script"],
        "indic": detected["indic"],
        "source": source,
        "note": note,
    }


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------

def _ass_time(t: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc"""
    hours = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int(t % 60)
    cs = int((t - int(t)) * 100)
    return f"{hours}:{mins:02d}:{secs:02d}.{cs:02d}"


def _ass_color(ass_bgr: str) -> str:
    """Convert &H00BBGGRR to ASS V4+ style color (already in ASS format)."""
    return ass_bgr


def _escape_ass_text(text: str) -> str:
    """Escape special characters for ASS format."""
    text = text.replace('\\', '\\\\')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('\n', '\\N')
    return text


def _lyrics_scale() -> float:
    """Global lyric size multiplier (bigger, bolder captions by default)."""
    try:
        value = float(os.environ.get("BEATSYNC_LYRICS_SCALE", "1.3"))
    except ValueError:
        value = 1.3
    return max(1.0, min(2.5, value))


# Vertical placement bands, from lowest to highest on screen. Consecutive lines
# are placed in different bands so the text moves with the cut instead of
# sitting in the same spot for the whole video.
_LYRIC_BANDS = ("bottom", "lower", "center", "upper", "top")
_BAND_WEIGHTS = {"bottom": 3, "lower": 3, "center": 2, "upper": 1, "top": 1}


def _band_anchor(band: str, width: int, height: int, margin_v: int) -> Tuple[int, int, int]:
    """Return (ASS alignment, x, y) for a placement band."""
    cx = width // 2
    if band == "top":
        return 8, cx, max(48, margin_v)
    if band == "upper":
        return 8, cx, int(height * 0.22)
    if band == "center":
        return 5, cx, height // 2
    if band == "lower":
        return 2, cx, int(height * 0.72)
    return 2, cx, height - max(48, margin_v)


def _smart_band_plan(count: int, seed: int = 0) -> List[str]:
    """Deterministic band per line that never repeats the previous band."""
    rng = random.Random(20240 + seed)
    plan: List[str] = []
    prev = None
    for _ in range(count):
        choices = [b for b in _LYRIC_BANDS if b != prev]
        pick = rng.choices(choices, weights=[_BAND_WEIGHTS[b] for b in choices])[0]
        plan.append(pick)
        prev = pick
    return plan


def generate_ass_file(
    lines: List[LyricLine],
    combo: FontCombo,
    output_path: str,
    video_width: int = 1920,
    video_height: int = 1080,
    fps: float = 30.0,
) -> str:
    """Generate an ASS subtitle file for the given lyrics and font combo.

    Creates two styles (primary + secondary) and renders each lyric line
    with the selected animation. Lines alternate between primary and
    secondary styling for visual variety, are scaled up for impact, and are
    placed in rotating vertical bands so captions move across the frame
    instead of staying pinned to one spot.
    """
    scale = _lyrics_scale()
    bands = _smart_band_plan(len(lines), seed=video_width)

    # Use transliterated text if available, else original
    display_lines = []
    for i, line in enumerate(lines):
        text = line.transliterated or line.text
        is_primary = (i % 2 == 0)  # Alternate primary/secondary
        display_lines.append((line.start_time, line.end_time, text, is_primary, bands[i]))

    # Build ASS header
    header = f"""[Script Info]
Title: BeatSync Lyrics
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""

    p = combo.primary
    s = combo.secondary

    p_bold = -1 if p.bold else 0
    p_italic = -1 if p.italic else 0
    s_bold = -1 if s.bold else 0
    s_italic = -1 if s.italic else 0

    # Scale sizes and beef up outline/shadow so big captions stay legible
    # over busy footage.
    p_size = int(round(p.size * scale))
    s_size = int(round(s.size * scale))
    p_outline = max(p.outline_width, int(round(p.outline_width * 1.4)))
    s_outline = max(s.outline_width, int(round(s.outline_width * 1.4)))
    p_shadow = max(p.shadow, 2)
    s_shadow = max(s.shadow, 2)

    header += f"Style: Primary,{_font_name(p.font_file)},{p_size},{_ass_color(p.primary_color)},{_ass_color(p.primary_color)},{_ass_color(p.outline_color)},{_ass_color('&H80000000')},{p_bold},{p_italic},0,0,100,100,1,0,1,{p_outline},{p_shadow},2,{p.margin_l},{p.margin_r},{p.margin_v},1\n"
    header += f"Style: Secondary,{_font_name(s.font_file)},{s_size},{_ass_color(s.primary_color)},{_ass_color(s.primary_color)},{_ass_color(s.outline_color)},{_ass_color('&H80000000')},{s_bold},{s_italic},0,0,100,100,1,0,1,{s_outline},{s_shadow},2,{s.margin_l},{s.margin_r},{s.margin_v},1\n"

    # Build events
    events = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    for start, end, text, is_primary, band in display_lines:
        if not text.strip():
            continue
        style = "Primary" if is_primary else "Secondary"
        dur = end - start

        base_mv = p.margin_v if is_primary else s.margin_v
        align, px, py = _band_anchor(band, video_width, video_height, base_mv)
        style_obj = p if is_primary else s
        anim_text = _apply_ass_animation(
            text, combo.animation, dur, fps, alignment=align, x=px, y=py,
            primary_color=style_obj.primary_color,
        )

        events += f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{anim_text}\n"

    # Background box style if enabled — draw behind each line at its band.
    if combo.bg_box:
        bg_events = ""
        for start, end, text, is_primary, band in display_lines:
            if not text.strip():
                continue
            mv = p.margin_v if is_primary else s.margin_v
            sz = p_size if is_primary else s_size
            align, px, py = _band_anchor(band, video_width, video_height, mv)
            top = max(0, py - sz - 12)
            bottom = min(video_height, py + 14)
            bg_events += (
                f"Dialogue: -1,{_ass_time(start)},{_ass_time(end)},Primary,,0,0,0,,"
                f"{{\\1c{combo.bg_color}\\p1}}m 0 {top} l {video_width} {top} "
                f"l {video_width} {bottom} l 0 {bottom}{{\\p0}}\n"
            )
        events = bg_events + events

    full_content = header + events

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_content)

    return output_path


def _font_name(font_file: str) -> str:
    """Extract a readable font name from filename.

    FFmpeg's subtitles filter uses fontconfig to find fonts by name.
    We use the filename without extension as the font name, and also
    register the font directory via the `fontsdir` option.
    """
    name = os.path.splitext(font_file)[0]
    # Remove weight/style suffixes for fontconfig matching
    name = re.sub(r'[-_]?(Regular|Bold|Italic|BoldItalic|Book|Medium|Light|Black|SemiBold|ExtraBold|ExtraLight|Thin|Oblique)$', '', name, flags=re.IGNORECASE)
    return name


# Bright gold used as the karaoke highlight; base text keeps the style colour.
_KARAOKE_HIGHLIGHT = "&H0000D4D4"


def _apply_ass_animation(text: str, animation: str, duration: float, fps: float,
                         alignment: int = 2, x: int = 0, y: int = 0,
                         primary_color: str = "&H00FFFFFF") -> str:
    """Apply ASS animation tags to text based on the animation type.

    ``alignment``/``x``/``y`` position the line at a chosen on-screen band so
    captions can move across the frame. The alignment override is emitted on
    every line so placement is independent of the base style.

    ``primary_color`` is the style colour, used as the un-highlighted base for
    word-level karaoke.
    """
    escaped = _escape_ass_text(text)
    half_dur = max(0.2, duration * 0.15)  # fade duration
    fade_ms = int(half_dur * 1000)
    dur_ms = max(200, int(duration * 1000))
    pos = f"\\an{alignment}\\pos({x},{y})"

    if animation == "slide_up":
        return f"{{{pos}\\move({x},{y + 40},{x},{y},{fade_ms})\\fad({fade_ms},{fade_ms})}}{escaped}"
    if animation == "slide_left":
        return f"{{{pos}\\move({x - 100},{y},{x},{y},{fade_ms})\\fad({fade_ms},{fade_ms})}}{escaped}"
    if animation == "pop":
        return f"{{{pos}\\fscx0\\fscy0\\t(0,{fade_ms},\\fscx100\\fscy100)\\fad({fade_ms},{fade_ms})}}{escaped}"
    if animation == "bounce":
        # Scale up with overshoot, then settle — a springy entrance.
        over = int(fade_ms * 0.6)
        return (f"{{{pos}\\fscx50\\fscy50\\fad({fade_ms},{fade_ms})"
                f"\\t(0,{over},\\fscx112\\fscy112)"
                f"\\t({over},{fade_ms},\\fscx100\\fscy100)}}{escaped}")
    if animation == "glow":
        # Soft pulsing blur so the line visibly breathes.
        return (f"{{{pos}\\blur2\\fad({fade_ms},{fade_ms})"
                f"\\t(0,{dur_ms // 2},\\blur10)\\t({dur_ms // 2},{dur_ms},\\blur2)}}{escaped}")
    if animation == "wave":
        # Per-character rocking motion with a staggered phase (a wave).
        chars = list(text)
        if not chars:
            return f"{{{pos}\\fad({fade_ms},{fade_ms})}}{escaped}"
        step = max(30, dur_ms // (len(chars) * 3))
        result = f"{{{pos}\\fad({fade_ms},{fade_ms})}}"
        for i, ch in enumerate(chars):
            esc_ch = _escape_ass_text(ch)
            ang = -18 if i % 2 == 0 else 18
            t0 = i * step
            result += (f"{{\\frz{ang}\\t({t0},{t0 + step},"
                       f"\\frz0\\fry{ang})\\t({t0 + step},{t0 + 2 * step},\\fry0)}}{esc_ch}")
        return result
    if animation == "karaoke":
        chars = list(text)
        if not chars:
            return f"{{{pos}\\fad({fade_ms},{fade_ms})}}{escaped}"
        char_dur = duration * 1000 / max(1, len(chars))
        result = f"{{{pos}}}"
        for ch in chars:
            esc_ch = _escape_ass_text(ch)
            result += f"{{\\k{int(char_dur / 10)}}}{esc_ch}"
        return result
    if animation == "karaoke_word":
        # Word-level karaoke: one \k per word so the highlight lands on whole
        # words rather than sweeping across characters. Sung text takes the
        # highlight colour, un-sung text the style colour.
        words = text.split()
        if not words:
            return f"{{{pos}\\fad({fade_ms},{fade_ms})}}{escaped}"
        total_cs = max(1, int(duration * 100))
        per = max(1, total_cs // len(words))
        result = f"{{{pos}\\1c{_KARAOKE_HIGHLIGHT}\\2c{primary_color}}}"
        for i, word in enumerate(words):
            esc_word = _escape_ass_text(word)
            k = per if i < len(words) - 1 else max(1, total_cs - per * (len(words) - 1))
            result += f"{{\\k{k}}}{esc_word}"
            if i < len(words) - 1:
                result += " "
        return result
    if animation == "typewriter":
        chars = list(text)
        if not chars:
            return f"{{{pos}\\fad({fade_ms},{fade_ms})}}{escaped}"
        char_delay = max(20, int(duration * 1000 / (len(chars) * 3)))
        result = f"{{{pos}}}"
        for i, ch in enumerate(chars):
            esc_ch = _escape_ass_text(ch)
            result += f"{{\\alpha&HFF&\\t({i * char_delay},{i * char_delay + 50},\\alpha&H00&)}}{esc_ch}"
        return result
    # Default: fade in place at the chosen band.
    return f"{{{pos}\\fad({fade_ms},{fade_ms})}}{escaped}"


# ---------------------------------------------------------------------------
# FFmpeg subtitle burn-in
# ---------------------------------------------------------------------------

def burn_lyrics_into_video(
    video_file: str,
    ass_file: str,
    output_file: str,
    fonts_dir: str = FONTS_DIR,
    use_nvenc: bool = False,
    gpu_encoder: str = 'h264_nvenc',
    fps: float = 30.0,
) -> str:
    """Burn ASS subtitles (lyrics) into a video file using FFmpeg.

    Uses the subtitles filter with libass for high-quality text rendering
    with custom fonts, outlines, and animations.
    """
    import ffmpeg_processing as fp

    # Escape path for FFmpeg filter (Windows-style escaping on all platforms)
    ass_escaped = ass_file.replace('\\', '/').replace(':', '\\:')

    # Build the subtitles filter with fontsdir
    filter_str = f"subtitles='{ass_escaped}':fontsdir='{fonts_dir.replace('\\', '/')}'"

    cmd = [
        fp.FFMPEG_PATH, '-nostdin', '-hide_banner',
        '-i', video_file,
        '-vf', filter_str,
        '-c:v', 'libx264' if not use_nvenc else gpu_encoder,
    ]

    if use_nvenc:
        cmd.extend(fp.get_nvenc_clip_quality_args(gpu_encoder, include_pix_fmt=True))
    else:
        cmd.extend(fp.get_clip_h264_quality_args(include_pix_fmt=True))

    # Copy audio stream as-is
    cmd.extend(['-c:a', 'copy', '-r', str(fps), '-fps_mode', 'cfr',
                '-movflags', '+faststart', '-y', output_file])

    print(f"   🎤 Burning lyrics into video...")
    print(f"      ASS file: {os.path.basename(ass_file)}")
    print(f"      Fonts dir: {fonts_dir}")

    result = fp._run_media_command(cmd, timeout=1800)

    if result.returncode != 0:
        err = fp._short_ffmpeg_error(result.stderr)
        print(f"   ⚠️  Lyrics burn-in failed ({err}); output without lyrics.")
        # Copy original as fallback
        import shutil
        shutil.copy2(video_file, output_file)
        return output_file

    print(f"   ✓ Lyrics burned in successfully")
    return output_file


# ---------------------------------------------------------------------------
# Full lyrics pipeline
# ---------------------------------------------------------------------------

def process_lyrics(
    lyrics_text: str,
    audio_duration: float,
    beat_times: Optional[List[float]] = None,
    font_combo_name: str = "",
    font_combo_index: int = -1,
    video_width: int = 1920,
    video_height: int = 1080,
    fps: float = 30.0,
    output_dir: str = "/tmp",
    do_transliterate: bool = True,
) -> Tuple[str, List[LyricLine], FontCombo]:
    """Full lyrics processing pipeline.

    Returns (ass_file_path, lyric_lines, font_combo).
    """
    # Parse lyrics
    lines = parse_lyrics(lyrics_text, audio_duration, beat_times)
    if not lines:
        return "", [], get_combo(0)

    # Transliterate if needed
    if do_transliterate:
        lines = apply_transliteration(lines)

    # Select font combo
    if font_combo_name:
        combo = get_combo_by_name(font_combo_name)
    elif font_combo_index >= 0:
        combo = get_combo(font_combo_index)
    else:
        combo = get_combo(0)

    # Generate ASS file
    ass_path = os.path.join(output_dir, f"lyrics_{int(time.time())}.ass")
    generate_ass_file(lines, combo, ass_path, video_width, video_height, fps)

    translit_count = sum(1 for l in lines if l.transliterated != l.text)
    print(f"   🎤 Lyrics processed: {len(lines)} lines, {translit_count} transliterated")
    print(f"   🎨 Font combo: {combo.name}")
    print(f"   📝 ASS file: {ass_path}")

    return ass_path, lines, combo


# ---------------------------------------------------------------------------
# Smart transition plan using preset
# ---------------------------------------------------------------------------

def build_smart_transitions_with_preset(
    profiles: List[Dict],
    preset: TransitionPreset,
    rng: random.Random,
) -> Tuple[List[str], List[float]]:
    """Build per-cut transition types and durations using a TransitionPreset.

    Returns (transition_types, transition_durations).
    """
    from smart_sync import compute_segment_energy

    transition_types: List[str] = []
    transition_durations: List[float] = []

    for i in range(len(profiles) - 1):
        seg_energy = compute_segment_energy(profiles[i])
        next_energy = compute_segment_energy(profiles[i + 1])

        # Determine context
        section_type = profiles[i].get("section_type", "body")
        target = profiles[i].get("target", "flow")
        next_target = profiles[i + 1].get("target", "flow")
        energy_delta = next_energy - seg_energy

        if section_type == "intro":
            context = "intro"
        elif section_type == "outro":
            context = "outro"
        elif next_target == "drop" and energy_delta > 0.15:
            context = "drop"
        elif target == "build" or section_type in {"bridge", "hook"}:
            context = "build"
        elif seg_energy >= 0.65 and next_energy >= 0.65:
            context = "high"
        elif seg_energy <= 0.35 and next_energy <= 0.35:
            context = "low"
        else:
            context = "medium"

        t_type = preset.pick(context, seg_energy, next_energy, i, rng)
        t_dur = preset.duration_for(seg_energy, next_energy)

        transition_types.append(t_type)
        transition_durations.append(t_dur)

    return transition_types, transition_durations
