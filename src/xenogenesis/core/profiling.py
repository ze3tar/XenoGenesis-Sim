"""Lightweight profiling helpers."""
import contextlib
import time
from typing import Iterator


@contextlib.contextmanager
def timer(name: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[PROFILE] {name}: {elapsed:.4f}s")
