"""Tests for the smart_sync scene-snap and multicam-group extensions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import smart_sync


def test_snap_to_scene_boundary_within_tolerance():
    assert smart_sync.snap_to_scene_boundary(5.2, [0.0, 5.0, 9.0]) == 5.0


def test_snap_to_scene_boundary_outside_tolerance_keeps_original():
    assert smart_sync.snap_to_scene_boundary(5.4, [0.0, 5.0, 9.0]) == 5.0
    # > tolerance from any boundary -> unchanged
    assert smart_sync.snap_to_scene_boundary(7.0, [0.0, 5.0, 9.0]) == 7.0


def test_snap_handles_empty_and_clamps_at_zero():
    assert smart_sync.snap_to_scene_boundary(3.0, []) == 3.0
    # A near boundary on the other side of zero snaps to the clamp, not negative.
    assert smart_sync.snap_to_scene_boundary(0.1, [-0.3]) == 0.0
    # A far boundary leaves the value untouched.
    assert smart_sync.snap_to_scene_boundary(0.1, [-2.0]) == 0.1


def test_assign_camera_groups_by_offset():
    files = ["/a.mp4", "/b.mp4", "/c.mp4"]
    groups = smart_sync.assign_camera_groups(files, {"/a.mp4": 0.0, "/b.mp4": 0.0004, "/c.mp4": 0.5})
    assert groups["/a.mp4"] == groups["/b.mp4"]      # same camera
    assert groups["/c.mp4"] != groups["/a.mp4"]      # different camera


def test_assign_camera_groups_unknown_without_offsets():
    groups = smart_sync.assign_camera_groups(["/a.mp4", "/b.mp4"], None)
    assert groups["/a.mp4"] == "cam@unknown"
