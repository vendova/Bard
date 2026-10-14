"""Shared helpers: config loading, paths, licence gate, ffmpeg wrappers."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


class MissingAssetError(RuntimeError):
    """Raised when a required asset/tool/font is missing (fail loudly)."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    root = project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    if not cfg_path.exists():
        raise MissingAssetError(f"config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(root)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the project root."""
    return (project_root() / cfg["paths"][key]).resolve()


# --------------------------------------------------------------------------- #
# Tool checks (Stage 1, Section 1.6 — fail loudly)
# --------------------------------------------------------------------------- #
def require_ffmpeg(min_major: int = 6) -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise MissingAssetError(
            "ffmpeg/ffprobe not found on PATH. Install FFmpeg >= "
            f"{min_major} (e.g. `apt-get install ffmpeg`)."
        )
    out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout
    first = out.splitlines()[0] if out else ""
    major = 0
    for tok in first.split():
        if tok[:1].isdigit():
            major = int(tok.split(".")[0])
            break
    if major < min_major:
        raise MissingAssetError(
            f"FFmpeg {major} found but >= {min_major} required: {first}"
        )
    return ffmpeg, ffprobe


def require_fonts(cfg: dict) -> dict[str, Path]:
    """Ensure every candidate font exists, else stop (Section 1.6)."""
    fonts_dir = resolve(cfg, "fonts_dir")
    wanted = (
        list(cfg["fonts"]["script_candidates"])
        + list(cfg["fonts"]["key_candidates"])
        + [cfg["fonts"]["watermark_font"]]
    )
    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in wanted:
        p = fonts_dir / name
        if p.exists():
            found[name] = p
        else:
            missing.append(name)
    if missing:
        raise MissingAssetError(
            "Missing fonts: " + ", ".join(missing)
            + f"\nRun `bash scripts/fetch_fonts.sh` (looks in {fonts_dir})."
        )
    return found


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr[-2000:]}"
        )
    return res


# --------------------------------------------------------------------------- #
# Licence gate (Section 1.1)
# --------------------------------------------------------------------------- #
@dataclass
class Licence:
    asset_id: str
    path: str
    kind: str
    licence: str
    source: str
    owner: str
    allowed: bool


def load_licences(cfg: dict) -> dict[str, Licence]:
    manifest = resolve(cfg, "licence_manifest")
    if not manifest.exists():
        raise MissingAssetError(f"licence manifest not found: {manifest}")
    rows: dict[str, Licence] = {}
    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(
            line for line in fh if not line.lstrip().startswith("#")
        )
        for r in reader:
            if not r.get("asset_id"):
                continue
            allowed = str(r.get("allowed", "")).strip().lower() in {"yes", "true", "1"}
            lic = Licence(
                asset_id=r["asset_id"].strip(),
                path=(r.get("path") or "").strip(),
                kind=(r.get("kind") or "").strip(),
                licence=(r.get("licence") or "").strip(),
                source=(r.get("source") or "").strip(),
                owner=(r.get("owner") or "").strip(),
                allowed=allowed,
            )
            rows[str(Path(lic.path).resolve())] = lic
            rows[lic.asset_id] = lic
    return rows


def licence_for(cfg: dict, licences: dict[str, Licence], asset_path: str | os.PathLike) -> Licence:
    """Return the licence row for an asset, or raise with the reason."""
    key = str(Path(asset_path).resolve())
    row = licences.get(key) or licences.get(os.path.basename(str(asset_path)))
    if row is None:
        raise PermissionError(
            f"REFUSING to render: no licence row for '{asset_path}'. "
            f"Add it to {cfg['paths']['licence_manifest']} with allowed=yes."
        )
    if not row.allowed:
        raise PermissionError(
            f"REFUSING to render '{asset_path}': licence='{row.licence}' "
            f"marked not allowed. Only use assets you have the right to publish."
        )
    return row


def gate_assets(cfg: dict, assets: list[str], kind: str) -> list[str]:
    """Filter a list of assets through the licence gate; report rejects."""
    licences = load_licences(cfg)
    ok, rejected = [], []
    for a in assets:
        try:
            licence_for(cfg, licences, a)
            ok.append(a)
        except PermissionError as exc:
            rejected.append(str(exc))
    if rejected:
        print("⚠️  Licence gate rejected:", file=sys.stderr)
        for r in rejected:
            print("   -", r, file=sys.stderr)
    return ok


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def write_json(path: str | os.PathLike, data) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def read_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dirs(cfg: dict) -> None:
    for key in ("work_dir", "renders_dir", "reports_dir"):
        resolve(cfg, key).mkdir(parents=True, exist_ok=True)
