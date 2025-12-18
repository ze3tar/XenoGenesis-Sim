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


@dataclass(frozen=True)
class KernelCache:
    """Cached kernel buffers for both isotropic and directional convolutions."""

    kernel: np.ndarray
    kernel_fft: np.ndarray
    grad_fft: tuple[np.ndarray, np.ndarray]
    positive_mass: float
    negative_mass: float


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
    """Construct a normalized, signed multi-ring kernel.

    Positive and negative bands are normalized independently to stabilize
    excitatory/inhibitory interactions and keep the kernel mass conserving.
    """

    kernel = np.zeros((size, size), dtype=np.float32)
    for (r_in, r_out), w in zip(rings, weights):
        ring = _radial_disk(size, r_out) - _radial_disk(size, r_in)
        kernel += w * ring
    pos = float(kernel[kernel > 0].sum())
    neg = float(-kernel[kernel < 0].sum())
    if pos > 0:
        kernel[kernel > 0] /= pos
    if neg > 0:
        kernel[kernel < 0] /= neg
    mass = float(np.abs(kernel).sum())
    if mass > 0:
        kernel /= mass
    return kernel


@lru_cache(maxsize=32)
def kernel_bank(params: KernelParams) -> KernelCache:
    """Return a multi-ring kernel, its FFT, and directional gradients."""

    kernel = multi_ring_kernel(params.size, params.rings, params.ring_weights)
    kernel_fft = np.fft.rfftn(kernel)
    grad_y, grad_x = np.gradient(kernel)
    grad_fft = (np.fft.rfftn(grad_y), np.fft.rfftn(grad_x))
    pos = float(kernel[kernel > 0].sum())
    neg = float(-kernel[kernel < 0].sum())
    return KernelCache(kernel=kernel, kernel_fft=kernel_fft, grad_fft=grad_fft, positive_mass=pos, negative_mass=neg)
