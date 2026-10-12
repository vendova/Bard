"""ISO base media file format (MP4/MOV/M4V) box/atom parser.

Reads the container structure directly from the bytes rather than trusting any
external tool. This is what lets the rest of the forensics package reason about
*how* a file was produced:

* which muxer/editor wrote the file (``ftyp`` brand, ``udta``/``meta`` tags),
* where the encoder put keyframes (``stss`` sync samples) and how frames are
  grouped (``stts``/``ctts``),
* edit lists (``elst``) — trims/offsets the editor imposed on the timeline.

Only the container is parsed here; decoding frames is left to ffmpeg.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Full boxes hold a 4-byte version/flags field + payload.
_FULL_BOXES = {
    "meta", "hdlr", "stsd", "stts", "ctts", "stsc", "stsz", "stss",
    "stco", "co64", "elst", "mvhd", "tkhd", "mdhd", "smhd", "vmhd",
    "dref", "url ", "nmhd", "gmhd", "keys", "stsh",
}
# Containers whose payload is more boxes (not raw data).
_CONTAINERS = {
    "moov", "trak", "mdia", "minf", "stbl", "edts", "dinf", "udta",
    "meta", "ilst", "moof", "traf", "mvex", "mfra", "sinf", "schi",
}

_HEADER = struct.Struct(">I4s")          # size + type
_HEADER_LARGE = struct.Struct(">Q4s")    # 64-bit size + type

# iTunes-style metadata boxes (``©too`` = encoder, ``©nam`` = title, ...) hold a
# child ``data`` box, so they must be recursed into as containers.
_ITUNES_TAGS = {
    "\xa9too", "\xa9nam", "\xa9ART", "\xa9alb", "\xa9day", "\xa9cmt",
    "\xa9gen", "\xa9wrt", "\xa9swr", "aART", "covr", "desc", "ldes",
    "tmpo", "trkn", "disk", "gnre", "catg", "keyw", "purl", "egid",
    "----", "mean", "name", "data",
}


def _is_container(btype: str) -> bool:
    return btype in _CONTAINERS or btype in _ITUNES_TAGS


@dataclass
class Box:
    """One parsed box: its type, payload range and immediate children."""
    type: str
    offset: int
    size: int
    header_size: int
    children: List["Box"] = field(default_factory=list)
    # Byte range of the payload (excludes the header).
    payload_start: int = 0
    payload_end: int = 0

    def find(self, path: str) -> Optional["Box"]:
        """Find a box by slash-separated path, e.g. ``moov/trak/mdia``."""
        node: Optional[Box] = self
        for part in path.split("/"):
            if node is None:
                return None
            node = next((c for c in node.children if c.type == part), None)
        return node

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def _read_boxes(data: bytes, start: int, end: int, depth: int = 0) -> List[Box]:
    boxes: List[Box] = []
    pos = start
    while pos + 8 <= end:
        size, btype = _HEADER.unpack_from(data, pos)
        header_size = 8
        if size == 1:
            if pos + 16 > end:
                break
            size = _HEADER_LARGE.unpack_from(data, pos)[0]
            header_size = 16
        elif size == 0:
            size = end - pos  # box extends to end of file/parent
        if size < header_size or pos + size > end:
            break  # malformed / truncated

        btype_s = btype.decode("latin-1")
        payload_start = pos + header_size
        payload_end = pos + size
        box = Box(type=btype_s, offset=pos, size=size, header_size=header_size,
                  payload_start=payload_start, payload_end=payload_end)

        child_start = payload_start
        if btype_s in _FULL_BOXES:
            child_start += 4  # skip version + flags
        if _is_container(btype_s) and depth < 10 and child_start < payload_end:
            box.children = _read_boxes(data, child_start, payload_end, depth + 1)
        boxes.append(box)
        pos += size
    return boxes


def parse_boxes(data: bytes) -> List[Box]:
    """Parse the top-level box tree of an ISO-BMFF file."""
    if len(data) < 8:
        return []
    return _read_boxes(data, 0, len(data))


# ---------------------------------------------------------------------------
# Typed accessors for the boxes we care about
# ---------------------------------------------------------------------------

def _brands(data: bytes, ftyp: Box) -> Tuple[str, List[str], List[str]]:
    p = ftyp.payload_start
    major = data[p:p + 4].decode("latin-1", "replace")
    minor = struct.unpack_from(">I", data, p + 4)[0] if p + 8 <= ftyp.payload_end else 0
    compatible: List[str] = []
    q = p + 8
    while q + 4 <= ftyp.payload_end:
        compatible.append(data[q:q + 4].decode("latin-1", "replace"))
        q += 4
    return major, compatible, []


def _box_text(data: bytes, box: Box) -> str:
    return data[box.payload_start:box.payload_end].decode("latin-1", "replace").strip("\x00")


_STRUCTURAL = {"hdlr", "ilst", "keys", "meta", "mdta", "free", "skip"}


def _data_string(data: bytes, data_box: Box) -> str:
    """Decode an iTunes ``data`` box value (skips its type/locale prefix)."""
    payload = data[data_box.payload_start:data_box.payload_end]
    if len(payload) > 8 and payload[4:8] == b"\x00\x00\x00\x00":
        return payload[8:].decode("utf-8", "replace").strip("\x00")
    return payload.decode("utf-8", "replace").strip("\x00")


def _string_tags(root: Box, data: bytes) -> Dict[str, str]:
    """Collect string metadata from ``moov/udta`` and QuickTime ``meta/ilst``."""
    tags: Dict[str, str] = {}
    for holder in ("moov/udta", "moov/udta/meta", "moov/meta"):
        node = root.find(holder)
        if not node:
            continue
        for child in node.walk():
            if child is node or child.type in _STRUCTURAL:
                continue
            data_box = child.find("data")
            if data_box is not None:  # iTunes-style item
                value = _data_string(data, data_box)
                if value and child.type not in tags:
                    tags[child.type] = value
            elif not child.children:  # plain freeform text box
                txt = _box_text(data, child)
                if txt and txt.isprintable() and child.type not in tags:
                    tags[child.type] = txt
    return tags


def _elst_entries(data: bytes, elst: Box) -> List[Tuple[int, int]]:
    """Return ``(segment_duration, media_time)`` pairs from an edit list."""
    p = elst.payload_start
    if p + 8 > elst.payload_end:
        return []
    version = data[p]
    count = struct.unpack_from(">I", data, p + 4)[0]
    entries: List[Tuple[int, int]] = []
    q = p + 8
    if version == 1:
        for _ in range(count):
            if q + 20 > elst.payload_end:
                break
            seg_dur, media_time = struct.unpack_from(">QQ", data, q)
            entries.append((seg_dur, media_time))
            q += 20
    else:
        for _ in range(count):
            if q + 12 > elst.payload_end:
                break
            seg_dur, media_time = struct.unpack_from(">II", data, q)
            entries.append((seg_dur, media_time))
            q += 12
    return entries


# An offset of a few frames is the normal B-frame reorder delay; anything past
# this many seconds means the timeline genuinely skips into the media.
_TRIM_OFFSET_SECONDS = 0.5


def edit_list_summary(data: bytes, root: Box) -> Tuple[bool, bool]:
    """Return ``(has_edit_list, indicates_trim)``.

    ffmpeg writes a benign single-entry edit list to compensate for B-frame
    reorder delay: a small positive ``media_time`` (a few frames) at the start.
    A *stream-copy* cut (keyframe-aligned, no re-encode) instead leaves a large
    ``media_time`` offset into the media, which is the container-level trace of
    an edit performed on an existing encoded stream. Note this does **not** fire
    for a trim that was re-encoded, because the re-encode re-zeroes the offset.
    """
    moov = root.find("moov")
    if moov is None:
        return False, False

    found = False
    for trak in moov.children:
        if trak.type != "trak":
            continue
        elst = trak.find("edts/elst")
        if elst is None:
            continue
        found = True
        entries = _elst_entries(data, elst)
        if not entries:
            continue
        mdhd = trak.find("mdia/mdhd")
        timescale = 1000
        if mdhd is not None:
            timescale, _ = _coerce_timescale_duration(data, mdhd)
        for _dur, media_time in entries:
            if media_time <= 0:
                continue
            if timescale and media_time / timescale > _TRIM_OFFSET_SECONDS:
                return True, True
    return found, False


def _has_edit_list(root: Box) -> bool:
    return root.find("moov/trak/edts/elst") is not None


def _stts_frame_count(data: bytes, stts: Box) -> int:
    """Sum sample counts in a time-to-sample box."""
    p = stts.payload_start + 4  # version/flags
    if p + 4 > stts.payload_end:
        return 0
    entries = struct.unpack_from(">I", data, p)[0]
    total = 0
    q = p + 4
    for _ in range(entries):
        if q + 8 > stts.payload_end:
            break
        count, _delta = struct.unpack_from(">II", data, q)
        total += count
        q += 8
    return total


def _sync_sample_count(data: bytes, stss: Box) -> int:
    p = stss.payload_start + 4
    if p + 4 > stss.payload_end:
        return 0
    return struct.unpack_from(">I", data, p)[0]


def _keyframe_indices(data: bytes, stss: Box) -> List[int]:
    p = stss.payload_start + 4
    if p + 4 > stss.payload_end:
        return []
    entries = struct.unpack_from(">I", data, p)[0]
    idxs: List[int] = []
    q = p + 4
    for _ in range(entries):
        if q + 4 > stss.payload_end:
            break
        idxs.append(struct.unpack_from(">I", data, q)[0])
        q += 4
    return idxs


def _coerce_timescale_duration(data: bytes, mdhd: Box) -> Tuple[int, int]:
    p = mdhd.payload_start
    version = data[p] if p < mdhd.payload_end else 0
    if version == 1:
        # version/flags(4) + creation(8) + modification(8) + timescale(4) + duration(8)
        q = p + 4 + 8 + 8
        timescale = struct.unpack_from(">I", data, q)[0] if q + 4 <= mdhd.payload_end else 0
        duration = struct.unpack_from(">Q", data, q + 4)[0] if q + 12 <= mdhd.payload_end else 0
    else:
        q = p + 4 + 4 + 4
        timescale = struct.unpack_from(">I", data, q)[0] if q + 4 <= mdhd.payload_end else 0
        duration = struct.unpack_from(">I", data, q + 4)[0] if q + 8 <= mdhd.payload_end else 0
    return timescale, duration


def _stco_offsets(data: bytes, stco: Box, large: bool) -> List[int]:
    """Return chunk offsets (stco = 32-bit, co64 = 64-bit)."""
    p = stco.payload_start + 4
    if p + 4 > stco.payload_end:
        return []
    count = struct.unpack_from(">I", data, p)[0]
    offsets: List[int] = []
    q = p + 4
    step = 8 if large else 4
    for _ in range(count):
        if q + step > stco.payload_end:
            break
        offsets.append(struct.unpack_from(">Q" if large else ">I", data, q)[0])
        q += step
    return offsets


def _stsz_sizes(data: bytes, stsz: Box) -> Tuple[int, int, List[int]]:
    """Return (sample_size, sample_count, sample_sizes)."""
    p = stsz.payload_start + 4
    if p + 8 > stsz.payload_end:
        return 0, 0, []
    sample_size, sample_count = struct.unpack_from(">II", data, p)
    if sample_size != 0:
        return sample_size, sample_count, [sample_size] * sample_count
    sizes: List[int] = []
    q = p + 8
    for _ in range(sample_count):
        if q + 4 > stsz.payload_end:
            break
        sizes.append(struct.unpack_from(">I", data, q)[0])
        q += 4
    return 0, sample_count, sizes


@dataclass
class ContainerInfo:
    major_brand: str = ""
    compatible_brands: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    box_types: List[str] = field(default_factory=list)
    box_count: int = 0
    tracks: int = 0
    has_edit_list: bool = False
    edit_list_is_trim: bool = False
    # Video specifics (first video track).
    timescale: int = 0
    duration: int = 0
    sample_count: int = 0
    sync_sample_count: int = 0
    keyframe_indices: List[int] = field(default_factory=list)
    chunk_offsets: List[int] = field(default_factory=list)
    sample_sizes: List[int] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.duration / self.timescale if self.timescale else 0.0

    def keyframe_times(self, frame_durations: List[float] | None = None,
                       fps: float = 0.0) -> List[float]:
        """Approximate keyframe timestamps (seconds).

        Uses per-sample durations when available, else a constant ``fps``.
        """
        out: List[float] = []
        if frame_durations:
            for idx in self.keyframe_indices:
                if 1 <= idx <= len(frame_durations):
                    out.append(sum(frame_durations[:idx - 1]))
        elif fps > 0:
            out = [(idx - 1) / fps for idx in self.keyframe_indices if idx >= 1]
        return [round(t, 4) for t in out]


def parse_container(data: bytes) -> ContainerInfo:
    """Parse *data* and summarise the containers, tags, tracks and keyframes."""
    root_boxes = parse_boxes(data)
    info = ContainerInfo()

    all_boxes: List[Box] = []
    for rb in root_boxes:
        all_boxes.extend(rb.walk())
    info.box_count = len(all_boxes)
    seen: List[str] = []
    for b in all_boxes:
        if b.type not in seen:
            seen.append(b.type)
    info.box_types = seen

    ftyp = next((b for b in root_boxes if b.type == "ftyp"), None)
    if ftyp:
        info.major_brand, info.compatible_brands, _ = _brands(data, ftyp)

    root = Box(type="root", offset=0, size=len(data), header_size=0,
               children=root_boxes)
    info.tags = _string_tags(root, data)
    info.has_edit_list, info.edit_list_is_trim = edit_list_summary(data, root)

    traks = [b for b in all_boxes if b.type == "trak"]
    info.tracks = len(traks)

    for trak in traks:
        stss = trak.find("mdia/minf/stbl/stss")
        stts = trak.find("mdia/minf/stbl/stts")
        mdhd = trak.find("mdia/mdhd")
        if stss is None and stts is None:
            continue  # not a timed media track we can summarise
        info.sync_sample_count = _sync_sample_count(data, stss) if stss else 0
        info.keyframe_indices = _keyframe_indices(data, stss) if stss else []
        info.sample_count = _stts_frame_count(data, stts) if stts else 0
        if mdhd:
            info.timescale, info.duration = _coerce_timescale_duration(data, mdhd)
        stsz = trak.find("mdia/minf/stbl/stsz")
        if stsz:
            _, _, info.sample_sizes = _stsz_sizes(data, stsz)
        stco = trak.find("mdia/minf/stbl/stco")
        co64 = trak.find("mdia/minf/stbl/co64")
        if stco:
            info.chunk_offsets = _stco_offsets(data, stco, large=False)
        elif co64:
            info.chunk_offsets = _stco_offsets(data, co64, large=True)
        break  # first summarisable (video) track

    return info
