"""Generate PLACEHOLDER captions aligned to a plan (for testing/demo only).

Real edits use authored captions (assets/captions/*.captions). This helper makes
a caption line for every captioned slot so a render can be exercised end to end
without you having to hand-time placeholder text.

    python src/make_placeholder_captions.py --plan out/reports/plan_x.json \
        --out assets/captions/placeholder.captions
"""
from __future__ import annotations

from pathlib import Path

from util import load_config, read_json, resolve

# Generic, non-copyrighted placeholder words (you own these).
SCRIPT = ["Stay with", "Every", "Hold me", "Never", "In your", "Feel the",
          "Dance with", "All night", "You and", "Forever", "Come back", "Stay"]
KEY = ["ME", "NIGHT", "CLOSE", "GO", "EYES", "LIGHT", "ME", "LONG", "I",
       "YOURS", "HOME", "ME"]


def generate(plan: dict, cfg: dict) -> list[str]:
    captioned = set(cfg["structure"]["captioned_sections"])
    lines = [
        "# PLACEHOLDER captions (self-authored) generated from a plan.",
        "# Replace with your own timed lyrics for a real edit.",
    ]
    k = 0
    for s in plan["shots"]:
        if s["section"] not in captioned:
            continue
        t = s["timeline_in"] + min(0.4, s["length"] * 0.25)
        lines.append(f"{t:05.2f} | {SCRIPT[k % len(SCRIPT)]} | {KEY[k % len(KEY)]}")
        k += 1
    return lines


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Placeholder captions from a plan")
    ap.add_argument("--config", default=None)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    plan = read_json(args.plan)
    lines = generate(plan, cfg)
    out = Path(args.out) if args.out else (resolve(cfg, "captions_dir") / "placeholder.captions")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ {len(lines) - 2} placeholder captions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
