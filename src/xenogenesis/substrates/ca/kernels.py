"""Kernel construction and caching for continuous CA stepping."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class KernelParams:
    size: int
    inner_radius: float
    outer_radius: float
    ring_ratio: float


def _radial_disk(size: int, radius: float) -> np.ndarray:
    grid = np.indices((size, size)).astype(np.float32)
    center = (size - 1) / 2.0
    dist2 = (grid[0] - center) ** 2 + (grid[1] - center) ** 2
    mask = dist2 <= radius * radius
    disk = mask.astype(np.float32)
    norm = disk.sum()
    if norm > 0:
        disk /= norm
    return disk


@lru_cache(maxsize=32)
def kernel_bank(params: KernelParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return inner/outer kernels and their FFTs for the given parameters.

    The ring kernel follows the native stepper semantics: density is taken from
    the annulus bounded by ``sqrt(inner_radius^2 * ring_ratio)`` and
    ``outer_radius``.
    """
    inner = _radial_disk(params.size, params.inner_radius)
    ring_inner_radius = np.sqrt(params.inner_radius * params.inner_radius * params.ring_ratio)
    outer = _radial_disk(params.size, params.outer_radius) - _radial_disk(params.size, ring_inner_radius)
    inner_fft = np.fft.rfftn(inner)
    outer_fft = np.fft.rfftn(outer)
    return inner, outer, inner_fft, outer_fft
