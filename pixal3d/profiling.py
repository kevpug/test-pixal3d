"""
Per-stage wall-clock timing.

A run has half a dozen very different stages — background removal, camera
estimation, four denoising cascades, VAE decode, mesh cleanup, UV unwrap,
texture bake — and on a slow GPU they are not slow in the proportions you would
guess. Without measurements, tuning is a coin flip.

Enable with ``PIXAL3D_TIMING=1`` (or ``--timing``); a summary prints at the end.
Timing synchronises the device around each stage, so leave it off for
benchmarking the last few percent.
"""

from typing import *
import os
import time
from contextlib import contextmanager


__all__ = ['enabled', 'set_enabled', 'stage', 'record', 'summary', 'reset']


_records: List[Tuple[str, float]] = []
_enabled: Optional[bool] = None


def enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get('PIXAL3D_TIMING', '0') == '1'
    return _enabled


def set_enabled(value: bool) -> None:
    global _enabled
    _enabled = value


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def stage(name: str):
    """Time a block and print it as it finishes."""
    if not enabled():
        yield
        return
    _sync()
    started = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        record(name, time.perf_counter() - started)


def record(name: str, seconds: float) -> None:
    _records.append((name, seconds))
    if enabled():
        print(f"[timing] {name}: {seconds:.1f}s")


def reset() -> None:
    _records.clear()


def summary() -> str:
    """A breakdown of everything recorded, slowest first."""
    if not _records:
        return ""
    total = sum(seconds for _, seconds in _records)
    width = max(len(name) for name, _ in _records)
    lines = ["", "Stage timings", "-" * 13]
    for name, seconds in sorted(_records, key=lambda r: -r[1]):
        share = seconds / total * 100 if total else 0
        lines.append(f"  {name.ljust(width)}  {seconds:7.1f}s  {share:5.1f}%")
    lines.append(f"  {'total'.ljust(width)}  {total:7.1f}s")
    return "\n".join(lines)


def print_summary() -> None:
    text = summary()
    if text:
        print(text)
