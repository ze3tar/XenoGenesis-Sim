import pytest

pytest.importorskip("numpy")
import numpy as np

from xenogenesis.substrates.ca.ca_model import CAStepper
from xenogenesis.substrates.ca.kernels import KernelParams
from xenogenesis.substrates.ca.genome import CAParams


def test_ca_step_equivalence():
    stepper = CAStepper()
    params = KernelParams(size=8, rings=((1.0, 3.0), (3.0, 4.0)), ring_weights=(1.0, -0.5))
    state = np.ones((8, 8), dtype=np.float32) * 0.5
    ca_params = CAParams(
        grid_size=8,
        kernel_params=params,
        mu=0.15,
        sigma=0.015,
        dt=0.1,
        growth_alpha=0.9,
        decay_lambda=0.0,
        regen_rate=0.05,
        consumption_rate=0.02,
        resource_diffusion=0.0,
        biomass_diffusion=0.0,
        noise_std=0.0,
        polarity_gain=0.3,
        polarity_decay=0.94,
        polarity_mobility=0.05,
        polarity_noise=0.0,
        max_mass=0.8,
        death_factor=0.5,
        elongation_trigger=1.3,
        fission_assist=0.0,
        render_gamma=1.0,
        contour_level=0.5,
        dominant_band=0,
        maintenance_cost=0.05,
        competition_scale=0.2,
        competition_radius=3.0,
        resource_capacity=1.2,
        resource_gradient=0.2,
        polarity_diffusion=0.02,
        polarity_mutation=0.02,
        directional_gain=1.0,
        division_threshold=0.5,
        division_fraction=0.5,
        reproduction_cost=0.15,
        resource_affinity=0.3,
        toxin_rate=0.01,
        drift_rate=0.0,
    )
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    a = stepper._step_numpy(state, ca_params, rng_a).state
    b = stepper.step(state, ca_params, rng=rng_b).state
    assert a.shape[0] == 4
    assert b.shape == a.shape
    assert np.allclose(a, b, atol=1e-4)
    assert np.all(np.isfinite(a))
