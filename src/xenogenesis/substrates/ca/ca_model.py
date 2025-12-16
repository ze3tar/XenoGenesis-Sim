"""CA stepping utilities with native fallback."""
from __future__ import annotations
import importlib
import importlib.util
import numpy as np
from typing import Optional

_native_spec = importlib.util.find_spec("xenogenesis_native")
if _native_spec:
    native_ca_step = importlib.import_module("xenogenesis_native").ca_step  # type: ignore
    HAS_NATIVE = True
else:
    native_ca_step = None
    HAS_NATIVE = False


def radial_kernel(size: int, radius: float) -> np.ndarray:
    grid = np.indices((size, size)).astype(np.float32)
    center = (size - 1) / 2.0
    dist = np.sqrt((grid[0] - center) ** 2 + (grid[1] - center) ** 2)
    ker = (dist <= radius).astype(np.float32)
    if ker.sum() > 0:
        ker /= ker.sum()
    return ker


class CAStepper:
    def __init__(self):
        self._use_native = HAS_NATIVE

    def step(self, state: np.ndarray, *, mu: float, sigma: float, dt: float, inner_radius: float, outer_radius: float, ring_ratio: float) -> np.ndarray:
        if self._use_native:
            return native_ca_step(state.astype(np.float32), mu, sigma, dt, inner_radius, outer_radius, ring_ratio)
        return self._step_numpy(state.astype(np.float32), mu, sigma, dt, inner_radius, outer_radius, ring_ratio)

    @staticmethod
    def _step_numpy(state: np.ndarray, mu: float, sigma: float, dt: float, inner_radius: float, outer_radius: float, ring_ratio: float) -> np.ndarray:
        size = state.shape[0]
        # Simple convolution via FFT fallback
        km = radial_kernel(size, inner_radius)
        kn = radial_kernel(size, outer_radius) - radial_kernel(size, inner_radius * ring_ratio)
        km_fft = np.fft.rfftn(km)
        kn_fft = np.fft.rfftn(kn)
        s_fft = np.fft.rfftn(state)
        m = np.fft.irfftn(s_fft * km_fft, s=state.shape).real
        n = np.fft.irfftn(s_fft * kn_fft, s=state.shape).real
        g = np.exp(-((n - mu) ** 2) / (2 * sigma * sigma))
        updated = np.clip(state + dt * (2 * g - 1), 0.0, 1.0)
        return updated.astype(np.float32)
