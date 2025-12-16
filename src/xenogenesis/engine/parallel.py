"""Parallel evaluation helpers."""
from __future__ import annotations
from multiprocessing import Pool
from typing import Callable, Iterable, Any


def parallel_map(fn: Callable[[Any], Any], items: Iterable[Any], workers: int):
    if workers <= 1:
        return list(map(fn, items))
    with Pool(processes=workers) as pool:
        return pool.map(fn, items)
