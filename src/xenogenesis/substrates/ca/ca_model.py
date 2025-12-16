"""CA stepping utilities with native fallback and FFT caching."""
from __future__ import annotations
import importlib
import importlib.util
from dataclasses import dataclass
import numpy as np

from .kernels import KernelParams, kernel_bank

_native_spec = importlib.util.find_spec("xenogenesis_native")
if _native_spec:
    native_ca_step = importlib.import_module("xenogenesis_native").ca_step  # type: ignore
    HAS_NATIVE = True
else:
    native_ca_step = None
    HAS_NATIVE = False


@dataclass
class StepStats:
    mass: float
    entropy: float
    edge_density: float


class CAStepper:
    """Stepper that prioritizes the native extension but keeps a fast FFT fallback."""

    def __init__(self):
        self._use_native = HAS_NATIVE

    def step(self, state: np.ndarray, *, mu: float, sigma: float, dt: float, inner_radius: float, outer_radius: float, ring_ratio: float) -> np.ndarray:
        if self._use_native:
            return native_ca_step(state.astype(np.float32), mu, sigma, dt, inner_radius, outer_radius, ring_ratio)
        return self._step_numpy(state.astype(np.float32), mu, sigma, dt, inner_radius, outer_radius, ring_ratio)

    @staticmethod
    def _step_numpy(state: np.ndarray, mu: float, sigma: float, dt: float, inner_radius: float, outer_radius: float, ring_ratio: float) -> np.ndarray:
        size = state.shape[0]
        params = KernelParams(size=size, inner_radius=inner_radius, outer_radius=outer_radius, ring_ratio=ring_ratio)
        _, _, km_fft, kn_fft = kernel_bank(params)
        s_fft = np.fft.rfftn(state)
        m = np.fft.irfftn(s_fft * km_fft, s=state.shape).real
        n = np.fft.irfftn(s_fft * kn_fft, s=state.shape).real
        g = np.exp(-((n - mu) ** 2) / (2 * sigma * sigma))
        updated = np.clip(state + dt * (2 * g - 1), 0.0, 1.0)
        return updated.astype(np.float32)
