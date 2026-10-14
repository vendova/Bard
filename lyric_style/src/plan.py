"""Stage 4 — Planner: cut list on the bar grid, shot assignment, captions.

Writes plan.json (every shot: source, source in/out, timeline in/out, section,
caption, placement anchor) and prints a text timeline mirroring Section 3.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from util import MissingAssetError, read_json, resolve, write_json


# --------------------------------------------------------------------------- #
# Beat grid + cut list (Section 3.2, Section 11 skeleton hardened)
# --------------------------------------------------------------------------- #
def snap_to_grid(t: float, grid: np.ndarray, tol: float) -> float:
    i = int(np.argmin(np.abs(grid - t)))
    return float(grid[i]) if abs(grid[i] - t) <= tol else float(t)


def build_grid(beats: list[float], with_half: bool = True) -> np.ndarray:
    b = np.asarray(beats, dtype=float)
    if not with_half or b.size < 2:
        return b
    half = (b[:-1] + b[1:]) / 2.0
    grid = np.concatenate([b, half])
    return np.sort(grid)


def tempo_class(bpm: float, cfg: dict) -> str:
    for name, spec in cfg["tempo_classes"].items():
        if spec["bpm_min"] <= bpm <= spec["bpm_max"]:
            return name
    return "reference_like"


def build_cut_list(track: dict, cfg: dict, layout_name: str,
                   start: float | None = None, seed: int = 0,
                   fps: int | None = None) -> list[dict]:
    """Return shots with timeline in/out, snapped to the beat grid.

    When `fps` is given the boundaries are additionally quantised to whole
    frames so the assembled duration is exact (no per-shot rounding drift).
    """
    bpm = track["bpm"]
    bar = 60.0 / bpm * 4.0
    tclass = tempo_class(bpm, cfg)
    shot_len_override = cfg["tempo_classes"][tclass]["shot_len_bars"]
    layout = cfg["structure"]["layouts"][layout_name]
    tol = cfg["structure"]["snap_tolerance_s"]
    jitter = cfg["structure"]["jitter_bars"]

    if start is None:
        start = track.get("recommended_windows", [{}])[0].get("start", 0.0)
    beats = np.asarray(track["beats"], dtype=float)
    grid = build_grid(track["beats"])

    rng = random.Random(seed)
    # Snap tolerance: at least a half-beat, so bar-length targets jittered by
    # +/- jitter_bars still land back on the grid.
    snap_tol = max(tol * 3, 0.5 * 60.0 / bpm)
    t = snap_to_grid(start, grid, snap_tol)

    def quant(x: float) -> float:
        return round(x * fps) / fps if fps else x

    t = quant(t)
    shots = []
    for section, count, length_bars, end_bars in layout:
        # Tempo class can override Section A/C length to keep the reference feel.
        base_bars = shot_len_override if section in ("A", "C") else length_bars
        end = end_bars
        if end is not None and section in ("A", "C"):
            end = min(end, base_bars)
        for i in range(count):
            if end is not None:
                frac = i / max(1, count - 1)
                L = base_bars + (end - base_bars) * frac
            else:
                L = base_bars
            # Optional jitter, but always snap back onto the grid.
            if jitter > 0:
                L += rng.uniform(-jitter, jitter)
            target = t + L * bar
            t_next = snap_to_grid(target, grid, tol * 3)
            if fps:
                t_next = quant(t_next)
            if t_next <= t:
                t_next = t + max(0.2, L * bar)
                if fps:
                    t_next = quant(t_next)
            shots.append({
                "section": section,
                "timeline_in": round(t, 4),
                "timeline_out": round(t_next, 4),
                "length": round(t_next - t, 4),
                "snapped": abs(t_next - target) > 1e-6,
            })
            t = t_next
    return shots


# --------------------------------------------------------------------------- #
# Shot assignment (Section 4)
# --------------------------------------------------------------------------- #
def _candidate_pool(index: dict, section: str) -> list[dict]:
    pool = []
    for clip in index["clips"]:
        for w in clip["windows"]:
            if cfg_reject_text(w):
                continue
            pool.append({**w, "source_file": clip["path"],
                         "source_name": Path(clip["path"]).name,
                         "source_duration": clip["duration"],
                         "source_cuts": clip["scene_cuts"]})
    pool.sort(key=lambda w: w["scores"].get(section, 0.0), reverse=True)
    return pool


def cfg_reject_text(w: dict) -> bool:
    return bool(w.get("has_text"))


def assign_shots(shots: list[dict], index: dict, cfg: dict, seed: int) -> list[dict]:
    rng = random.Random(seed + 7)
    used_ids: set[str] = set()
    used_sources: set[str] = set()
    last_subject: str | None = None
    pools = {sec: _candidate_pool(index, sec)
             for sec in cfg["selection"]["section_energy"]}
    n_sources = len({c["path"] for c in index["clips"]})
    alternate = cfg["selection"]["alternate_subjects"] and n_sources > 1

    for shot in shots:
        sec = shot["section"]
        pool = pools[sec]

        def ok(cand, require_new_source: bool) -> bool:
            cid = f"{cand['source_file']}#{cand['window_start']}"
            if cid in used_ids:
                return False
            if require_new_source and cand["source_file"] in used_sources:
                return False
            if (alternate and last_subject is not None
                    and cand["source_name"] == last_subject):
                return False
            return True

        chosen = None
        # Preference order: new source + new window > new window > anything.
        for require_new in (True, False):
            for cand in pool:
                if ok(cand, require_new):
                    chosen = cand
                    break
            if chosen:
                break
        if chosen is None and pool:  # last resort: reuse least-recently
            chosen = pool[rng.randrange(len(pool))]
        if chosen is None:
            raise MissingAssetError(
                f"No candidate windows for section {sec}. Ingest more footage."
            )
        cid = f"{chosen['source_file']}#{chosen['window_start']}"
        used_ids.add(cid)
        used_sources.add(chosen["source_file"])
        last_subject = chosen["source_name"]
        shot.update({
            "source_file": chosen["source_file"],
            "source_name": chosen["source_name"],
            "source_in": chosen["window_start"],
            "source_out": chosen["window_end"],
            "score": chosen["scores"].get(sec, 0.0),
            "face_cx": chosen["face_cx"],
            "brightness": chosen["brightness"],
            "has_text": chosen["has_text"],
        })
    return shots


# --------------------------------------------------------------------------- #
# Captions (Section 6.7)
# --------------------------------------------------------------------------- #
def parse_captions(path: str) -> list[dict]:
    out = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        try:
            start = float(parts[0])
        except ValueError:
            continue
        out.append({"start": start, "script": parts[1], "key": parts[2].upper()})
    return out


def assign_captions(shots: list[dict], captions: list[dict], cfg: dict,
                    bar_s: float | None = None) -> list[dict]:
    """Attach supplied captions to captioned shots (Section 6.7).

    Preference: a caption whose start falls inside the shot; otherwise the
    nearest unused caption within one bar of the shot's midpoint. Captions are
    never invented — an unfilled captioned slot is left empty and flagged.
    """
    captioned = set(cfg["structure"]["captioned_sections"])
    if bar_s is None:
        bar_s = 2.0
    unused = list(captions)
    for shot in shots:
        shot["caption"] = None
        if shot["section"] not in captioned:
            continue
        inside = [c for c in unused
                  if shot["timeline_in"] <= c["start"] < shot["timeline_out"]]
        if inside:
            cap = min(inside, key=lambda c: c["start"])
        else:
            mid = (shot["timeline_in"] + shot["timeline_out"]) / 2.0
            near = [c for c in unused if abs(c["start"] - mid) <= bar_s]
            cap = min(near, key=lambda c: abs(c["start"] - mid)) if near else None
        if cap is not None:
            unused.remove(cap)
            shot["caption"] = {"script": cap["script"], "key": cap["key"],
                               "line_start": cap["start"]}
    return shots


def anchor_for(face_cx: float, cfg: dict) -> str:
    """Section 6.5: lockup hugs the side opposite the main face."""
    return "right" if face_cx < 0.5 else "left"


# --------------------------------------------------------------------------- #
# Plan + timeline
# --------------------------------------------------------------------------- #
def build_plan(cfg: dict, track: dict, index: dict, caption_path: str | None,
               layout_name: str, seed: int, window_start: float | None = None) -> dict:
    fps = cfg["output"]["fps"]
    shots = build_cut_list(track, cfg, layout_name, start=window_start, seed=seed,
                           fps=fps)

    # Keep the whole edit inside the track: shift the window earlier if the plan
    # would run past the audio, and drop trailing shots if it still cannot fit.
    dur = float(track.get("duration") or 0.0)
    if dur > 0 and shots:
        total = shots[-1]["timeline_out"] - shots[0]["timeline_in"]
        if window_start is None and shots[0]["timeline_in"] + total > dur + 1e-3:
            # Align the shifted window to the beat grid so cuts stay on the grid.
            grid = build_grid(track["beats"])
            new_start = float(np.clip(dur - total, 0.0, max(0.0, dur - total)))
            new_start = float(grid[int(np.argmin(np.abs(grid - new_start)))])
            shots = build_cut_list(track, cfg, layout_name, start=new_start,
                                   seed=seed, fps=fps)
        while shots and shots[-1]["timeline_out"] > dur + 1e-3:
            shots.pop()

    shots = assign_shots(shots, index, cfg, seed)
    captions = parse_captions(caption_path) if caption_path else []
    bar_s = 60.0 / track["bpm"] * 4.0
    shots = assign_captions(shots, captions, cfg, bar_s=bar_s)
    for s in shots:
        s["anchor"] = anchor_for(s.get("face_cx", 0.5), cfg)

    missing = [i for i, s in enumerate(shots)
               if s["section"] in cfg["structure"]["captioned_sections"]
               and s["caption"] is None]
    return {
        "layout": layout_name,
        "seed": seed,
        "bpm": track["bpm"],
        "tempo_class": tempo_class(track["bpm"], cfg),
        "window_start": shots[0]["timeline_in"] if shots else 0.0,
        "window_end": shots[-1]["timeline_out"] if shots else 0.0,
        "shots": shots,
        "missing_captions": missing,
        "_beats": track.get("beats", []),
    }


def format_timeline(plan: dict) -> str:
    lines = [f"Layout: {plan['layout']}  seed={plan['seed']}  "
             f"BPM={plan['bpm']}  class={plan['tempo_class']}",
             f"{'#':>2} {'in':>7} {'out':>7} {'len':>6} {'sec':>3} "
             f"{'text':>4} {'snap':>4}  caption / source"]
    for i, s in enumerate(plan["shots"], 1):
        cap = ""
        if s["caption"]:
            cap = f"{s['caption']['script']} + {s['caption']['key']}"
        lines.append(
            f"{i:>2} {s['timeline_in']:>7.2f} {s['timeline_out']:>7.2f} "
            f"{s['length']:>6.2f} {s['section']:>3} "
            f"{'yes' if s['caption'] else 'no':>4} "
            f"{'y' if s.get('snapped') else 'n':>4}  {cap:<26} {s.get('source_name','')}"
        )
    if plan["missing_captions"]:
        lines.append(f"⚠️  {len(plan['missing_captions'])} captioned slot(s) empty: "
                     f"{plan['missing_captions']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from util import load_config, require_ffmpeg
    ap = argparse.ArgumentParser(description="Stage 4: build a plan")
    ap.add_argument("--config", default=None)
    ap.add_argument("--track-analysis", required=True)
    ap.add_argument("--footage-index", default=None)
    ap.add_argument("--captions", default=None)
    ap.add_argument("--layout", default="reference")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--window-start", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    track = read_json(args.track_analysis)
    idx_path = args.footage_index or str(resolve(cfg, "footage_dir").parent / "footage_index.json")
    index = read_json(idx_path)
    caps = args.captions
    plan = build_plan(cfg, track, index, caps, args.layout, seed, args.window_start)
    out = args.out or str(resolve(cfg, "reports_dir") / f"plan_{args.layout}_{seed}.json")
    write_json(out, plan)
    print(format_timeline(plan))
    print(f"\n✅ plan -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
