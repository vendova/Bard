"""Attribute a media file to the software that wrote it, from container bytes.

Turns low-level container evidence into a human answer: "this file was written
by ffmpeg/libx264", "this carries an Adobe Premiere XMP packet", etc. It is
deliberately honest about confidence — a ``ftyp`` brand is strong evidence, a
generic ``Handler`` tag is weak — so the report does not overclaim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .containers import ContainerInfo

# brand -> (software, confidence)
_BRAND_HINTS: Dict[str, Tuple[str, str]] = {
    "isom": ("ISO Base Media (generic muxer, often ffmpeg)", "low"),
    "iso2": ("ISO Base Media v2", "low"),
    "iso4": ("ISO Base Media v4", "low"),
    "iso5": ("ISO Base Media v5", "low"),
    "iso6": ("ISO Base Media v6", "low"),
    "mp41": ("MP4 v1 (older muxer)", "low"),
    "mp42": ("MP4 v2 muxer (ffmpeg/HandBrake)", "medium"),
    "avc1": ("H.264 in MP4", "low"),
    "dash": ("DASH packager (e.g. gpac/MP4Box, Bento4)", "medium"),
    "qt  ": ("Apple QuickTime (or macOS/iOS writer)", "medium"),
    "M4V ": ("Apple iTunes/iOS (M4V)", "medium"),
    "M4A ": ("Apple iTunes/iOS audio", "medium"),
    "3gp4": ("3GPP mobile writer", "medium"),
    "3gp5": ("3GPP mobile writer", "medium"),
    "F4V ": ("Adobe Flash/F4V muxer", "medium"),
    "CAEP": ("Adobe Premiere/After Effects", "high"),
    "ccrf": ("Adobe Common Container", "high"),
    "MSNV": ("Sony PSP/Nero", "medium"),
    "XAVC": ("Sony XAVC camera", "high"),
    "MA1A": ("Apple ProRes authoring", "high"),
}

# (tag substring) -> (software, confidence)
_TAG_HINTS: List[Tuple[str, str, str]] = [
    ("lavf", "ffmpeg / libavformat", "high"),
    ("libavformat", "ffmpeg / libavformat", "high"),
    ("handbrake", "HandBrake", "high"),
    ("adobe", "Adobe (Premiere/Media Encoder)", "medium"),
    ("premiere", "Adobe Premiere Pro", "high"),
    ("after effects", "Adobe After Effects", "high"),
    ("quicktime", "Apple QuickTime", "medium"),
    ("davinci", "DaVinci Resolve", "high"),
    ("resolve", "DaVinci Resolve", "medium"),
    ("final cut", "Final Cut Pro", "high"),
    ("kdenlive", "Kdenlive (MLT)", "high"),
    ("mlt ", "MLT framework (Kdenlive/Shotcut)", "high"),
    ("shotcut", "Shotcut", "high"),
    ("capcut", "CapCut", "medium"),
    ("obs", "OBS Studio", "medium"),
    ("avassetwriter", "Apple AVAssetWriter (macOS/iOS)", "high"),
    ("wondershare", "Wondershare Filmora", "high"),
    ("capella", "Capella", "low"),
    ("sony", "Sony camera/tooling", "medium"),
    ("gopro", "GoPro", "medium"),
]

_TAG_KEYS = {"©too", "\xa9too", "too", "encoder", "\xa9swr", "swr",
             "Handler", "handler", "com.apple.quicktime.software",
             "com.apple.quicktime.author", "com.apple.quicktime.make",
             "com.apple.quicktime.model", "XMP", "xml "}


@dataclass
class Fingerprint:
    software: str = "unknown"
    confidence: str = "none"
    methods_used: List[str] = field(default_factory=list)
    editor_metadata: Dict[str, str] = field(default_factory=dict)
    brands: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "software": self.software,
            "confidence": self.confidence,
            "methods_used": self.methods_used,
            "editor_metadata": self.editor_metadata,
            "brands": self.brands,
            "notes": self.notes,
        }


_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def fingerprint(info: ContainerInfo) -> Fingerprint:
    """Combine container evidence into a best-guess writer attribution."""
    fp = Fingerprint()
    fp.brands = [info.major_brand] + list(info.compatible_brands)
    best: Tuple[str, str] = ("unknown", "none")

    for brand in fp.brands:
        hit = _BRAND_HINTS.get(brand.strip())
        if not hit:
            continue
        fp.methods_used.append(f"ftyp brand '{brand.strip()}'")
        if _RANK[hit[1]] > _RANK[best[1]]:
            best = hit

    for key, value in info.tags.items():
        lowered = value.lower()
        if key in _TAG_KEYS or "encoder" in key.lower():
            fp.editor_metadata[key] = value
        for needle, sw, conf in _TAG_HINTS:
            if needle in lowered:
                fp.methods_used.append(f"metadata '{key}' contains '{needle.strip()}'")
                if _RANK[conf] > _RANK[best[1]]:
                    best = (sw, conf)

    fp.software, fp.confidence = best
    if not fp.methods_used:
        fp.notes.append("No identifying brand or metadata tag found.")
    if fp.editor_metadata.get("©too") or fp.editor_metadata.get("encoder"):
        fp.methods_used.append("encoder tag present")

    return fp
