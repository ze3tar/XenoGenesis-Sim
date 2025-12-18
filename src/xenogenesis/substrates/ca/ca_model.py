"""CA stepping utilities with native fallback and FFT caching."""
from __future__ import annotations
import importlib
import importlib.util
from dataclasses import dataclass
import numpy as np

from .kernels import KernelParams, kernel_bank
from .genome import CAParams

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

    def step(self, state: np.ndarray, params: CAParams, *, rng: np.random.Generator | None = None) -> np.ndarray:
        if self._use_native:
            # Native kernels are deprecated for Lenia-style dynamics; fall back to NumPy.
            return self._step_numpy(state.astype(np.float32), params, rng)
        return self._step_numpy(state.astype(np.float32), params, rng)

    @staticmethod
    def _step_numpy(
        state: np.ndarray,
        params: CAParams,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if state.ndim == 2:
            biomass = state
            resource = np.ones_like(state)
            polarity_x = np.zeros_like(state)
            polarity_y = np.zeros_like(state)
            extras: list[np.ndarray] = []
        elif state.ndim == 3:
            channels = state.shape[0]
            biomass = state[0]
            resource = state[1] if channels > 1 else np.ones_like(biomass)
            polarity_x = state[2] if channels > 2 else np.zeros_like(biomass)
            polarity_y = state[3] if channels > 3 else np.zeros_like(biomass)
            extras = [state[i] for i in range(4, channels)]
        else:
            raise ValueError("state must be (H, W) or (C, H, W)")

        _, kernel_fft = kernel_bank(params.kernel_params)
        s_fft = np.fft.rfftn(biomass)
        activation = np.fft.irfftn(s_fft * kernel_fft, s=biomass.shape, axes=(0, 1)).real

        grad_y, grad_x = np.gradient(biomass)
        polarity_x = params.polarity_decay * polarity_x + params.polarity_gain * grad_x
        polarity_y = params.polarity_decay * polarity_y + params.polarity_gain * grad_y

        growth = params.growth_alpha * np.tanh((activation - params.mu) / max(params.sigma, 1e-6))
        growth_term = growth * resource

        resource += params.regen_rate * (1.0 - resource)
        resource -= params.consumption_rate * biomass * np.maximum(growth_term, 0.0)
        if params.resource_diffusion > 0:
            lap_r = (
                -4 * resource
                + np.roll(resource, 1, axis=0)
                + np.roll(resource, -1, axis=0)
                + np.roll(resource, 1, axis=1)
                + np.roll(resource, -1, axis=1)
            )
            resource += params.resource_diffusion * lap_r
        resource = np.clip(resource, 0.0, 1.0)
        biomass_decay = params.decay_lambda * biomass
        transport = params.polarity_mobility * (polarity_x + polarity_y)
        delta_biomass = params.dt * (growth_term - biomass_decay) + params.dt * transport
        if params.biomass_diffusion > 0:
            lap = (
                -4 * biomass
                + np.roll(biomass, 1, axis=0)
                + np.roll(biomass, -1, axis=0)
                + np.roll(biomass, 1, axis=1)
                + np.roll(biomass, -1, axis=1)
            )
            delta_biomass += params.biomass_diffusion * lap
        if params.fission_assist > 0:
            lap = (
                -4 * biomass
                + np.roll(biomass, 1, axis=0)
                + np.roll(biomass, -1, axis=0)
                + np.roll(biomass, 1, axis=1)
                + np.roll(biomass, -1, axis=1)
            )
            delta_biomass += params.fission_assist * lap
        biomass = biomass + delta_biomass
        high_density = biomass > params.max_mass
        biomass[high_density] *= params.death_factor

        rng = rng or np.random.default_rng()
        biomass += rng.normal(0, params.noise_std, biomass.shape)
        polarity_x += rng.normal(0, params.polarity_noise, biomass.shape)
        polarity_y += rng.normal(0, params.polarity_noise, biomass.shape)
        biomass = np.clip(biomass, 0.0, 1.0)
        polarity_x = np.clip(polarity_x, -1.0, 1.0)
        polarity_y = np.clip(polarity_y, -1.0, 1.0)

        return np.stack(
            (
                biomass.astype(np.float32),
                resource.astype(np.float32),
                polarity_x.astype(np.float32),
                polarity_y.astype(np.float32),
                *extras,
            )
        )
