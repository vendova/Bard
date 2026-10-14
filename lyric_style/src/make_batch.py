"""Batch controller (Section 8.4).

Samples variants (track, window, footage subset, caption set, layout, font pair,
seed), rejects footage overlap > 40% between variants, logs each variant's full
tuple to manifest.csv, and names files {track_slug}_{layout}_{seed}.mp4.

    python src/make_batch.py --config config.yaml --count 40 --min-distinct 0.6 \
        --out ./out/renders
"""
from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

from util import load_config, read_json, resolve, require_ffmpeg, write_json


def discover_tracks(cfg: dict) -> list[Path]:
    mdir = resolve(cfg, "music_dir")
    return [p for p in sorted(mdir.glob("*"))
            if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}]


def load_track_analysis(track: Path) -> dict | None:
    p = track.with_suffix(".track_analysis.json")
    return read_json(p) if p.exists() else None


def footage_groups(cfg: dict) -> dict[str, list[str]]:
    """Group footage by source prefix (one shoot/film per group)."""
    fdir = resolve(cfg, "footage_dir")
    groups: dict[str, list[str]] = {}
    for p in sorted(fdir.glob("*")):
        if p.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm", ".avi"}:
            continue
        key = p.stem.split("_")[0]
        groups.setdefault(key, []).append(str(p))
    return groups


def caption_for_track(cfg: dict, track: Path) -> str | None:
    """Match a caption file to a track by shared stem, else the only one."""
    cdir = resolve(cfg, "captions_dir")
    exact = cdir / f"{track.stem}.captions"
    if exact.exists():
        return str(exact)
    caps = [p for p in sorted(cdir.glob("*.captions"))]
    return str(caps[0]) if len(caps) == 1 else None


def overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def sample_variants(cfg: dict, count: int, min_distinct: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    tracks = discover_tracks(cfg)
    if not tracks:
        raise SystemExit("No music tracks found. Add tracks to assets/music.")
    analyses = {t: load_track_analysis(t) for t in tracks}
    analyses = {t: a for t, a in analyses.items() if a}
    if not analyses:
        raise SystemExit("No track_analysis.json found. Run src/analyze_audio.py first.")
    groups = footage_groups(cfg)
    layouts = cfg["batch"]["layouts"]

    # Cartesian of (track, window, layout) then randomise with a fixed seed.
    combos = []
    for t, a in analyses.items():
        for wi, w in enumerate(a.get("recommended_windows", [{"start": 0.0}])[:3]):
            for lay in layouts:
                combos.append((t, wi, w["start"], lay))
    rng.shuffle(combos)
    combos = list(itertools.cycle(combos)) if len(combos) < count else combos

    variants, seen_sets = [], []
    for combo in combos:
        if len(variants) >= count:
            break
        t, wi, start, lay = combo
        vseed = rng.randrange(2 ** 31)
        # Footage subset: single group, or two groups (B from a different one).
        keys = list(groups)
        if not keys:
            raise SystemExit("No footage groups. Add clips to assets/footage.")
        g1 = rng.choice(keys)
        subset = set(groups[g1])
        mixed = None
        if len(keys) > 1 and rng.random() < 0.3:
            g2 = rng.choice([k for k in keys if k != g1])
            mixed = set(groups[g2])
        fileset = subset | (mixed or set())
        if any(overlap(fileset, s) > cfg["selection"]["max_overlap_between_variants"]
               for s in seen_sets):
            continue
        seen_sets.append(fileset)
        variants.append({
            "track": str(t),
            "window_index": wi,
            "window_start": start,
            "layout": lay,
            "footage_subset": sorted(subset),
            "footage_mix": sorted(mixed) if mixed else [],
            "caption_file": caption_for_track(cfg, t),
            "font_pair": cfg["fonts"]["script"] + "|" + cfg["fonts"]["key"],
            "seed": vseed,
        })
    return variants


def slug(track: str) -> str:
    return Path(track).stem.replace(" ", "_").replace("/", "_")


def write_manifest(variants: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["index", "output", "track", "window_index", "window_start", "layout",
            "footage_subset", "footage_mix", "caption_file", "font_pair", "seed"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, v in enumerate(variants):
            out = f"{slug(v['track'])}_{v['layout']}_{v['seed']}.mp4"
            w.writerow([i, out, v["track"], v["window_index"], v["window_start"],
                        v["layout"], ";".join(map(str, v["footage_subset"])),
                        ";".join(map(str, v["footage_mix"])),
                        v["caption_file"] or "", v["font_pair"], v["seed"]])


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Batch: sample + render variants")
    ap.add_argument("--config", default=None)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--min-distinct", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--render", action="store_true",
                    help="actually render (default: plan only)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    count = args.count or cfg["batch"]["count"]
    min_distinct = args.min_distinct or cfg["batch"]["min_distinct"]
    variants = sample_variants(cfg, count, min_distinct, cfg["seed"])

    out_dir = Path(args.out) if args.out else resolve(cfg, "renders_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    write_manifest(variants, manifest)
    write_json(out_dir / "variants.json", variants)
    print(f"✅ {len(variants)} variants -> {manifest}")
    print(f"   distinct footage sets: {len({tuple(v['footage_subset']) for v in variants})}")
    print(f"   tracks used: {len({v['track'] for v in variants})}, "
          f"layouts: {sorted({v['layout'] for v in variants})}")

    if not args.render:
        print("   (dry run; pass --render to produce videos)")
        return 0

    from render import render_plan
    from plan import build_plan
    from qc import qc
    index = read_json(resolve(cfg, "footage_dir").parent / "footage_index.json")
    for i, v in enumerate(variants):
        track = read_json(Path(v["track"]).with_suffix(".track_analysis.json"))
        plan = build_plan(cfg, track, index, v["caption_file"], v["layout"],
                          v["seed"], window_start=v["window_start"])
        out_name = f"{slug(v['track'])}_{v['layout']}_{v['seed']}.mp4"
        video = render_plan(cfg, plan, v["track"], v["caption_file"], out_name)
        qc(cfg, plan, str(video))
        print(f"[{i + 1}/{len(variants)}] {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
