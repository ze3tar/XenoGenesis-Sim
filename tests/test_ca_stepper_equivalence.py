import pytest

pytest.importorskip("numpy")
import numpy as np

from xenogenesis.substrates.ca.ca_model import CAStepper
from xenogenesis.substrates.ca.kernels import KernelParams


def test_ca_step_equivalence():
    stepper = CAStepper()
    params = KernelParams(size=8, rings=((1.0, 3.0), (3.0, 4.0)), ring_weights=(1.0, -0.5))
    state = np.ones((8, 8), dtype=np.float32) * 0.5
    rng = np.random.default_rng(0)
    a = stepper._step_numpy(
        state,
        0.15,
        0.015,
        0.1,
        params,
        0.05,
        0.02,
        0.0,
        0.9,
        0.3,
        0.94,
        0.05,
        0.8,
        0.5,
        0.0,
        rng,
    )
    b = stepper.step(
        state,
        mu=0.15,
        sigma=0.015,
        dt=0.1,
        kernel_params=params,
        regen_rate=0.05,
        consumption_rate=0.02,
        noise_std=0.0,
        growth_alpha=0.9,
        polarity_gain=0.3,
        polarity_decay=0.94,
        polarity_mobility=0.05,
        max_mass=0.8,
        death_factor=0.5,
        polarity_noise=0.0,
        rng=np.random.default_rng(0),
    )
    assert a.shape[0] == 4
    assert b.shape == a.shape
    assert np.allclose(a, b, atol=1e-4)
    assert np.all(np.isfinite(a))
