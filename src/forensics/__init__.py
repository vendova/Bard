"""Offline media forensics: how a finished video was made.

* :mod:`forensics.containers` - ISO-BMFF box/atom parsing.
* :mod:`forensics.fingerprint` - fingerprint the writing software from the
  container brand and metadata tags.
* :mod:`forensics.reencode` - GOP/keyframe timeline and re-encode/splice traces.
* :mod:`forensics.scenes` - shot-boundary detection from decoded frames.

:func:`analyze_file` ties them together into a single report.
"""
__all__ = ["analyze_file", "format_report"]


def __getattr__(name: str):
    # Import lazily so `python -m forensics.report` does not load the
    # submodule twice (which emits a runpy RuntimeWarning).
    if name in __all__:
        from . import report
        return getattr(report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
