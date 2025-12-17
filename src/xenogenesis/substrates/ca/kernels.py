"""Kernel construction and caching for continuous CA stepping."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class KernelParams:
    """Parameters for band-pass, multi-ring kernels.

    Attributes
    ----------
    size:
        Kernel size (assumes square grid).
    rings:
        Sequence of ``(inner_radius, outer_radius)`` tuples defining annuli.
    ring_weights:
        Per-ring weights; positive for excitation, negative for inhibition.
    """

    size: int
    rings: tuple[tuple[float, float], ...]
    ring_weights: tuple[float, ...]


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


def multi_ring_kernel(size: int, rings: tuple[tuple[float, float], ...], weights: tuple[float, ...]) -> np.ndarray:
    """Construct a normalized multi-ring kernel.

    Positive and negative bands are normalized independently to stabilize
    excitatory/inhibitory interactions.
    """

    kernel = np.zeros((size, size), dtype=np.float32)
    for (r_in, r_out), w in zip(rings, weights):
        ring = _radial_disk(size, r_out) - _radial_disk(size, r_in)
        kernel += w * ring
    pos = kernel[kernel > 0].sum()
    neg = -kernel[kernel < 0].sum()
    if pos > 0:
        kernel[kernel > 0] /= pos
    if neg > 0:
        kernel[kernel < 0] /= neg
    return kernel


@lru_cache(maxsize=32)
def kernel_bank(params: KernelParams) -> tuple[np.ndarray, np.ndarray]:
    """Return a multi-ring kernel and its FFT for the given parameters."""

    kernel = multi_ring_kernel(params.size, params.rings, params.ring_weights)
    kernel_fft = np.fft.rfftn(kernel)
    return kernel, kernel_fft
