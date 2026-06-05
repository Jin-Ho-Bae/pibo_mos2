"""Memory management helpers for Colab's limited 12 GB RAM."""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc
from contextlib import contextmanager


def memory_usage_mb() -> float:
    """Resident-set memory in MB. Cross-platform best effort."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20
    except Exception:
        if sys.platform == "linux":
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        return float("nan")


def free_memory(verbose: bool = False) -> None:
    """Force garbage collection and clear matplotlib figures."""
    gc.collect()
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    if verbose:
        print(f"[mem] {memory_usage_mb():.1f} MB after GC")


@contextmanager
def track_memory(tag: str = ""):
    """Context manager for tracking peak memory inside a block."""
    tracemalloc.start()
    before = memory_usage_mb()
    try:
        yield
    finally:
        after = memory_usage_mb()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"[mem:{tag}] before={before:.1f} MB  "
            f"after={after:.1f} MB  peak_alloc={peak / 2 ** 20:.1f} MB"
        )


def truncate_history(history: list, max_len: int = 5000) -> list:
    """Drop the oldest entries when history grows beyond `max_len`."""
    if len(history) <= max_len:
        return history
    return history[-max_len:]
