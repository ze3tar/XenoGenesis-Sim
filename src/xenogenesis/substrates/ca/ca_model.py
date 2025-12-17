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
        kernel_params: KernelParams,
        regen_rate: float,
        consumption_rate: float,
        noise_std: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self._use_native:
            # Native kernels are deprecated for Lenia-style dynamics; fall back to NumPy.
            return self._step_numpy(
                state.astype(np.float32),
                mu,
                sigma,
                dt,
                kernel_params,
                regen_rate,
                consumption_rate,
                noise_std,
                rng,
            )
        return self._step_numpy(
            state.astype(np.float32),
            mu,
            sigma,
            dt,
            kernel_params,
            regen_rate,
            consumption_rate,
            noise_std,
            rng,
        )

    @staticmethod
    def _step_numpy(
        state: np.ndarray,
        mu: float,
        sigma: float,
        dt: float,
        kernel_params: KernelParams,
        regen_rate: float,
        consumption_rate: float,
        noise_std: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if state.ndim == 2:
            biomass = state
            resource = np.ones_like(state)
        elif state.shape[0] == 2:
            biomass, resource = state[0], state[1]
        else:
            raise ValueError("state must be (H, W) or (2, H, W)")

        _, kernel_fft = kernel_bank(kernel_params)
        s_fft = np.fft.rfftn(biomass)
        activation = np.fft.irfftn(s_fft * kernel_fft, s=biomass.shape, axes=(0, 1)).real

        dx = np.roll(biomass, -1, axis=1) - np.roll(biomass, 1, axis=1)
        dy = np.roll(biomass, -1, axis=0) - np.roll(biomass, 1, axis=0)
        motion_bias = 0.03 * (dx + dy)

        growth = 2.0 * np.exp(-0.5 * ((activation - mu) / sigma) ** 2) - 1.0
        growth_term = growth * resource

        resource += regen_rate * (1.0 - resource)
        resource -= consumption_rate * biomass * resource
        resource = np.clip(resource, 0.0, 1.0)

        delta_biomass = dt * growth_term + motion_bias
        biomass = biomass + delta_biomass
        high_density = biomass > 0.75
        biomass[high_density] *= 0.98

        rng = rng or np.random.default_rng()
        biomass += rng.normal(0, noise_std, biomass.shape)
        biomass = np.clip(biomass, 0.001, 1.0)

        return np.stack((biomass.astype(np.float32), resource.astype(np.float32)))
