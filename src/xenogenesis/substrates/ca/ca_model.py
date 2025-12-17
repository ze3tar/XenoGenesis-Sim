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
        # The NumPy path contains the most up-to-date biological dynamics, so use it
        # even when the native extension is available.
        self._use_native = False

    def step(
        self,
        state: np.ndarray,
        *,
        mu: float,
        sigma: float,
        dt: float,
        inner_radius: float,
        outer_radius: float,
        ring_ratio: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self._use_native:
            return native_ca_step(state.astype(np.float32), mu, sigma, dt, inner_radius, outer_radius, ring_ratio)
        return self._step_numpy(state.astype(np.float32), mu, sigma, dt, inner_radius, outer_radius, ring_ratio, rng)

    @staticmethod
    def _step_numpy(
        state: np.ndarray,
        mu: float,
        sigma: float,
        dt: float,
        inner_radius: float,
        outer_radius: float,
        ring_ratio: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        size = state.shape[0]
        params = KernelParams(size=size, inner_radius=inner_radius, outer_radius=outer_radius, ring_ratio=ring_ratio)
        _, _, km_fft, kn_fft = kernel_bank(params)
        s_fft = np.fft.rfftn(state)
        m = np.fft.irfftn(s_fft * km_fft, s=state.shape).real
        n = np.fft.irfftn(s_fft * kn_fft, s=state.shape).real
        dx = np.roll(state, -1, axis=1) - np.roll(state, 1, axis=1)
        dy = np.roll(state, -1, axis=0) - np.roll(state, 1, axis=0)
        motion_bias = 0.03 * (dx + dy)
        growth = np.exp(-((n - mu) ** 2) / (2 * sigma ** 2))
        delta = dt * (growth - state) + motion_bias
        state = state + delta
        high_density = state > 0.75
        state[high_density] *= 0.98
        rng = rng or np.random.default_rng()
        state += rng.normal(0, 0.002, state.shape)
        state = np.clip(state, 0.001, 1.0)
        return state.astype(np.float32)
