"""CA stepping utilities with native fallback and FFT caching."""
from __future__ import annotations
import importlib
import importlib.util
from dataclasses import dataclass
import numpy as np

from .kernels import KernelParams, KernelCache, kernel_bank
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
    reproduction_events: int
    resource: float


@dataclass
class StepResult:
    state: np.ndarray
    stats: StepStats


class CAStepper:
    """Stepper that prioritizes the native extension but keeps a fast FFT fallback."""

    def __init__(self):
        # The NumPy path contains the most up-to-date biological dynamics, so use it
        # even when the native extension is available.
        self._use_native = False
        self._resource_gradient_cache: dict[tuple[int, int], np.ndarray] = {}
        self._drift_phase: float = 0.0

    def step(self, state: np.ndarray, params: CAParams, *, rng: np.random.Generator | None = None) -> StepResult:
        if self._use_native:
            # Native kernels are deprecated for Lenia-style dynamics; fall back to NumPy.
            return self._step_numpy(state.astype(np.float32), params, rng)
        return self._step_numpy(state.astype(np.float32), params, rng)

    def _resource_gradient(self, shape: tuple[int, int]) -> np.ndarray:
        if shape in self._resource_gradient_cache:
            return self._resource_gradient_cache[shape]
        h, w = shape
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        gradient = (x + 0.6 * y) * 0.5
        self._resource_gradient_cache[shape] = gradient
        return gradient

    @staticmethod
    def _laplacian(field: np.ndarray) -> np.ndarray:
        return (
            -4 * field
            + np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        )

    @staticmethod
    def _entropy(field: np.ndarray) -> float:
        hist, _ = np.histogram(field, bins=32, range=(0, 1), density=True)
        hist = hist + 1e-9
        return float(-(hist * np.log2(hist)).sum())

    def _anisotropic_reproduction(
        self,
        biomass: np.ndarray,
        resource: np.ndarray,
        polarity_x: np.ndarray,
        polarity_y: np.ndarray,
        params: CAParams,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        division_mask = (biomass > params.division_threshold) & (resource > params.resource_affinity)
        if not np.any(division_mask):
            return biomass, resource, polarity_x, polarity_y, 0
        yy, xx = np.indices(biomass.shape, dtype=np.int32)
        direction = np.stack((polarity_y, polarity_x))
        norm = np.linalg.norm(direction, axis=0) + 1e-6
        direction_unit = direction / norm
        grad_y, grad_x = np.gradient(biomass)
        fallback_y = np.sign(grad_y)
        fallback_x = np.sign(grad_x)
        offset_y = np.rint(direction_unit[0]).astype(np.int32)
        offset_x = np.rint(direction_unit[1]).astype(np.int32)
        offset_y = np.where(offset_y == 0, fallback_y, offset_y)
        offset_x = np.where(offset_x == 0, fallback_x, offset_x)
        offset_y = np.clip(offset_y, -1, 1).astype(np.int32)
        offset_x = np.clip(offset_x, -1, 1).astype(np.int32)
        transfer = np.clip(params.division_fraction * biomass * division_mask, 0.0, biomass)
        biomass = np.maximum(biomass - transfer - params.reproduction_cost * division_mask, 0.0)
        target_y = (yy + offset_y) % biomass.shape[0]
        target_x = (xx + offset_x) % biomass.shape[1]
        offspring_biomass = np.zeros_like(biomass)
        np.add.at(offspring_biomass, (target_y, target_x), transfer)
        biomass = biomass + offspring_biomass

        offspring_px = np.zeros_like(polarity_x)
        offspring_py = np.zeros_like(polarity_y)
        np.add.at(offspring_px, (target_y, target_x), polarity_x * division_mask)
        np.add.at(offspring_py, (target_y, target_x), polarity_y * division_mask)
        mutation = rng.normal(0.0, params.polarity_mutation, size=biomass.shape)
        offspring_px += mutation * division_mask
        offspring_py += mutation * division_mask
        polarity_x = np.where(division_mask, offspring_px, polarity_x)
        polarity_y = np.where(division_mask, offspring_py, polarity_y)
        reproduction_events = int(np.count_nonzero(division_mask))
        return biomass, resource, polarity_x, polarity_y, reproduction_events

    def _step_numpy(
        self,
        state: np.ndarray,
        params: CAParams,
        rng: np.random.Generator | None = None,
    ) -> StepResult:
        rng = rng or np.random.default_rng()
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

        kernel_cache: KernelCache = kernel_bank(params.kernel_params)
        s_fft = np.fft.rfftn(biomass)
        activation = np.fft.irfftn(s_fft * kernel_cache.kernel_fft, s=biomass.shape, axes=(0, 1)).real
        directional = params.directional_gain * (
            np.fft.irfftn(s_fft * kernel_cache.grad_fft[1], s=biomass.shape, axes=(0, 1)).real * polarity_x
            + np.fft.irfftn(s_fft * kernel_cache.grad_fft[0], s=biomass.shape, axes=(0, 1)).real * polarity_y
        )
        kernel_response = activation + directional
        competition_kernel_fft = np.fft.rfftn(np.abs(kernel_cache.kernel))
        local_density = np.fft.irfftn(np.fft.rfftn(biomass) * competition_kernel_fft, s=biomass.shape, axes=(0, 1)).real
        competition_penalty = params.competition_scale * local_density

        grad_y, grad_x = np.gradient(biomass)
        polarity_x = params.polarity_decay * polarity_x + params.polarity_gain * grad_x
        polarity_y = params.polarity_decay * polarity_y + params.polarity_gain * grad_y
        if params.polarity_diffusion > 0:
            lap_px = self._laplacian(polarity_x)
            lap_py = self._laplacian(polarity_y)
            polarity_x += params.polarity_diffusion * lap_px
            polarity_y += params.polarity_diffusion * lap_py

        growth_signal = kernel_response - params.maintenance_cost - competition_penalty
        growth = params.growth_alpha * np.tanh(growth_signal / max(params.sigma, 1e-6))
        curvature = self._laplacian(biomass)
        split_force = np.clip(curvature, 0.0, None)
        if params.split_gain > 0:
            split_mask = biomass > params.split_threshold
            growth -= split_force * params.split_gain * split_mask
        resource_factor = np.clip(resource / params.resource_capacity, 0.0, 1.0)
        growth_term = growth * biomass * resource_factor

        resource += params.regen_rate * (params.resource_capacity - resource)
        gradient = self._resource_gradient(biomass.shape)
        self._drift_phase += params.drift_rate
        drift = np.sin(self._drift_phase) * gradient
        resource += params.resource_gradient * gradient + drift
        resource -= params.consumption_rate * np.maximum(growth_term, 0.0)
        resource -= params.toxin_rate * biomass
        if params.resource_diffusion > 0:
            resource += params.resource_diffusion * self._laplacian(resource)
        resource = np.clip(resource, 0.0, params.resource_capacity)

        biomass_decay = params.decay_lambda * biomass
        transport = params.polarity_mobility * (polarity_x * grad_x + polarity_y * grad_y)
        res_grad_y, res_grad_x = np.gradient(resource)
        normal_mag = np.hypot(grad_x, grad_y) + 1e-6
        normal_x = grad_x / normal_mag
        normal_y = grad_y / normal_mag
        directional_growth = res_grad_x * normal_x + res_grad_y * normal_y
        anisotropic_term = params.motility_gain * directional_growth

        delta_biomass = params.dt * (growth_term - biomass_decay + transport + anisotropic_term)
        if params.biomass_diffusion > 0:
            delta_biomass += params.biomass_diffusion * self._laplacian(biomass)
        if params.fission_assist > 0:
            delta_biomass += params.fission_assist * self._laplacian(biomass)
        biomass = biomass + delta_biomass
        high_density = biomass > params.max_mass
        biomass[high_density] *= params.death_factor

        energy_signal = growth_term - params.maintenance_cost * biomass - biomass_decay
        death_mask = (biomass < params.death_threshold) | (energy_signal < 0)
        if np.any(death_mask):
            biomass = np.where(death_mask, 0.0, biomass)
            polarity_x = np.where(death_mask, 0.0, polarity_x)
            polarity_y = np.where(death_mask, 0.0, polarity_y)

        # Enforce non-negative biomass before reproduction to preserve mass semantics.
        biomass = np.clip(biomass, 0.0, params.max_mass)

        biomass, resource, polarity_x, polarity_y, reproduction_events = self._anisotropic_reproduction(
            biomass, resource, polarity_x, polarity_y, params, rng
        )

        biomass += rng.normal(0, params.noise_std, biomass.shape)
        polarity_x += rng.normal(0, params.polarity_noise, biomass.shape)
        polarity_y += rng.normal(0, params.polarity_noise, biomass.shape)

        # HARD biological constraints
        biomass = np.clip(biomass, 0.0, params.max_mass)
        resource = np.clip(resource, 0.0, params.resource_capacity)
        polarity_x = np.clip(polarity_x, -1.0, 1.0)
        polarity_y = np.clip(polarity_y, -1.0, 1.0)

        stats = StepStats(
            mass=float(biomass.mean()),
            entropy=self._entropy(biomass),
            edge_density=float(np.mean(np.abs(np.gradient(biomass)))),
            reproduction_events=reproduction_events,
            resource=float(resource.mean()),
        )

        stacked = np.stack(
            (
                biomass.astype(np.float32),
                resource.astype(np.float32),
                polarity_x.astype(np.float32),
                polarity_y.astype(np.float32),
                *extras,
            )
        )
        return StepResult(state=stacked, stats=stats)
