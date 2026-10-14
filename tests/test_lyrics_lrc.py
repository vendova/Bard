"""Tests for lyric language detection and LRC output."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lyrics_engine as le
import transliteration as tl


INDIC_SAMPLES = {
    "devanagari": "तुम ही हो",
    "telugu": "నీ కళ్ళలో",
    "tamil": "கண்ணே கண்ணே",
    "malayalam": "കണ്ണേ",
    "kannada": "ಕಣ್ಣೇ",
    "bengali": "আমি তোমাকে",
    "gurmukhi": "ਤੂੰ ਹੀ",
    "gujarati": "તું જ છો",
    "odia": "ତୁମେ",
    "sinhala": "ඔබ මගේ",
}


@pytest.mark.parametrize("script,text", INDIC_SAMPLES.items())
def test_indic_scripts_detected(script, text):
    info = tl.detect_language(text)
    assert info["script"] == script
    assert info["indic"] is True
    assert info["needs_transliteration"] is True


@pytest.mark.parametrize("script,text", INDIC_SAMPLES.items())
def test_indic_scripts_transliterate_to_latin(script, text):
    out = tl.transliterate(text)
    assert out != text
    # No source-script characters should survive transliteration.
    assert not any(tl.needs_transliteration(ch) for ch in out)
    assert out.strip()


def test_english_is_latin_not_indic():
    info = tl.detect_language("You are my everything")
    assert info["script"] == "latin"
    assert info["indic"] is False
    assert info["needs_transliteration"] is False


def test_lrc_timestamp_formatting():
    assert le.lrc_timestamp(0) == "[00:00.00]"
    assert le.lrc_timestamp(65.25) == "[01:05.25]"
    # Rounding at .995 must roll over, not print .100
    assert le.lrc_timestamp(59.999) == "[01:00.00]"


def test_format_lrc_has_timestamps_and_metadata():
    lines = [le.LyricLine(text="hello", start_time=0.0, end_time=2.0),
             le.LyricLine(text="world", start_time=2.0, end_time=4.0)]
    out = le.format_lrc(lines, title="My Song")
    assert "[ti:My Song]" in out
    assert "[00:00.00]hello" in out
    assert "[00:02.00]world" in out


def test_build_lyrics_artifact_from_plain_text(tmp_path):
    art = le.build_lyrics_artifact("तुम ही हो\nमेरे सनम",
                                   audio_duration=30.0, output_dir=str(tmp_path))
    assert art["source"] == "lyrics"
    assert art["indic"] is True
    assert len(art["lines"]) == 2
    assert art["lrc_text"].count("[") >= 2
    assert Path(art["lrc_path"]).exists()
    # Transliteration is attached for non-Latin lines.
    assert art["lines"][0].transliterated


def test_build_lyrics_artifact_honours_pasted_lrc(tmp_path):
    pasted = "[00:10.00]first line\n[00:20.00]second line"
    art = le.build_lyrics_artifact(pasted, audio_duration=30.0,
                                   output_dir=str(tmp_path))
    starts = [round(l.start_time, 2) for l in art["lines"]]
    assert starts == [10.0, 20.0]


def test_build_lyrics_artifact_without_asr_reports_cleanly(tmp_path):
    # No lyrics and no ASR backend installed -> no crash, explanatory note.
    art = le.build_lyrics_artifact("", audio_duration=30.0,
                                   audio_file="/nonexistent.mp3",
                                   output_dir=str(tmp_path))
    assert art["lines"] == []
    assert art["lrc_text"] == ""
    assert "speech-to-text" in art["note"].lower() or "auto-transcription" in art["note"].lower()


def test_lines_from_segments_builds_timed_lines():
    lines = le.lines_from_segments([
        {"start": 1.0, "end": 3.0, "text": "alpha"},
        {"start": 3.0, "end": 5.0, "text": "  "},   # skipped
        {"start": 5.0, "end": 7.0, "text": "beta"},
    ])
    assert [l.text for l in lines] == ["alpha", "beta"]
    assert lines[0].start_time == 1.0
    assert lines[1].end_time == 7.0
