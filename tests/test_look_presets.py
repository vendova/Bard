"""Tests for cinematic looks, theme presets and the grade filter builder."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import look_presets as lp
import theme_presets as tp
from font_presets import get_combo_names
from transition_presets import get_preset_names
from video_effects import EffectsConfig, parse_effects_from_ui, build_clip_effect_filters
from media_paths import resolve_ffmpeg

FFMPEG, _ = resolve_ffmpeg()
requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and Path(FFMPEG).exists()), reason="ffmpeg not available"
)


def test_look_table_shape():
    assert len(lp.LOOK_PRESETS) >= 12
    names = lp.get_look_names()
    assert "None" in names
    # lookup helpers round-trip and fall back safely
    assert lp.get_look_by_name("Teal & Orange").name == "Teal & Orange"
    assert lp.get_look_by_name("does-not-exist").name == "None"
    assert lp.get_look(0).name == "None"


def test_all_looks_build_filters_without_error():
    for look in lp.LOOK_PRESETS:
        filters = lp.build_look_filters(look, (1920, 1080))
        assert isinstance(filters, list)


def test_letterbox_adds_pad_with_even_height():
    look = lp.get_look_by_name("Blockbuster")
    filters = lp.build_look_filters(look, (1920, 1080))
    pads = [f for f in filters if f.startswith("pad=")]
    assert pads, "Blockbuster should add letterbox bars"
    # pad height must stay even for yuv420p
    assert "pad=iw:ih+" in pads[0]
    delta = int(pads[0].split("ih+")[1].split(":")[0])
    assert delta % 2 == 0


def test_grain_and_halation_present_for_noir_look():
    noir = [l for l in lp.LOOK_PRESETS if l.name == "Neo Noir Night"][0]
    filters = lp.build_look_filters(noir, (1280, 720))
    joined = " ".join(filters)
    assert "noise=" in joined
    assert "blend=all_mode=screen" in joined


def test_missing_lut_is_ignored():
    look = lp.LookPreset(name="x", lut="definitely_not_here.cube")
    assert lp.resolve_lut("definitely_not_here.cube") is None
    assert not any("lut3d" in f for f in lp.build_look_filters(look, (640, 480)))


def test_effects_config_carries_look():
    cfg = parse_effects_from_ui(
        False, False, False, False, False, False, False, False, False,
        False, 2.0, "none", 0.35, look_name="Teal & Orange",
    )
    assert cfg.look_name == "Teal & Orange"
    # A look applies even with the effects master toggle off.
    filters, _ = build_clip_effect_filters(cfg, 0, 2.0, 30.0, (640, 480))
    assert any("colorbalance" in f or "eq=" in f for f in filters)


def test_no_look_no_effects_is_empty():
    cfg = EffectsConfig()
    filters, speed = build_clip_effect_filters(cfg, 0, 2.0, 30.0, (640, 480))
    assert filters == [] and speed is None


def test_theme_bindings_resolve():
    font_names = set(get_combo_names())
    trans_names = set(get_preset_names())
    look_names = set(lp.get_look_names())
    assert len(tp.THEME_PRESETS) >= 6
    for theme in tp.THEME_PRESETS:
        if theme.name == "None":
            continue
        assert theme.font_combo in font_names, theme
        assert theme.transition_preset in trans_names, theme
        assert theme.look in look_names, theme


def test_theme_helpers():
    assert tp.is_noop(tp.get_theme_by_name("None"))
    assert not tp.is_noop(tp.get_theme_by_name("Neo Noir"))
    assert tp.get_theme_by_name("nope").name == "None"
    assert tp.get_random_theme().name != "None"


def test_gradio_binding_matches_positional_args():
    """Gradio binds inputs positionally; a mismatch silently shifts arguments.

    process_video / regenerate_video must expose exactly as many positional
    parameters as they have bound inputs (regenerate/resume_job_id stay
    keyword-only because they are not bound to any UI control).
    """
    import inspect
    gui = pytest.importorskip("gui")
    app = gui.create_ui()
    # The two render buttons carry the shared inputs; the other deps are
    # theme change (10), load (0), forensics (2), etc.
    input_counts = sorted(
        len(d.get("inputs", [])) for d in app.config["dependencies"]
        if d.get("inputs")
    )
    n_shared = max(input_counts)
    for fn in (gui.process_video, gui.regenerate_video):
        sig = inspect.signature(fn)
        pos = [p for p in sig.parameters.values()
               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert len(pos) == n_shared, (
            f"{fn.__name__}: {len(pos)} positional params vs {n_shared} bound inputs")
        # Unbound control-flow params must be keyword-only.
        for name in ("regenerate", "resume_job_id"):
            if name in sig.parameters:
                assert sig.parameters[name].kind == sig.parameters[name].KEYWORD_ONLY


def test_apply_theme_fills_pickers():
    gui = pytest.importorskip("gui")
    # 'None' is a no-op: every output is an empty gr.update().
    noop = gui.apply_theme("None", "", False, False, False, False, 0.4, 1.3)
    assert all(u.get("value", None) is None for u in noop)

    theme = tp.get_theme_by_name("Neo Noir")
    vals = [u.get("value") for u in
            gui.apply_theme("Neo Noir", "", False, False, False, False, 0.4, 1.3)]
    assert vals[0] == "Neo Noir"
    assert vals[1] == theme.font_combo
    assert vals[2] == theme.look
    assert vals[3] == theme.transition_preset
    assert vals[4] == theme.effects_enabled


@requires_ffmpeg
def test_look_filters_render_in_ffmpeg(tmp_path):
    for name in ("Blockbuster", "Neo Noir Night", "Teal & Orange", "Noir"):
        filters = lp.build_look_filters(lp.get_look_by_name(name), (320, 240))
        chain = ",".join(["trim=duration=1", "setpts=PTS-STARTPTS",
                          "scale=320:240", "fps=25"] + filters)
        out = tmp_path / f"{name.replace(' ', '_').replace('&', 'and')}.mp4"
        res = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=25:duration=1",
             "-vf", chain, "-frames:v", "5", str(out)],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, f"{name}: {res.stderr}"
        assert out.exists() and out.stat().st_size > 0
