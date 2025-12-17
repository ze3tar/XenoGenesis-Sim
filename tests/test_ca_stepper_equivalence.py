import pytest

pytest.importorskip("numpy")
import numpy as np

from xenogenesis.substrates.ca.ca_model import CAStepper, HAS_NATIVE
from xenogenesis.substrates.ca.kernels import KernelParams


def test_ca_step_equivalence():
    stepper = CAStepper()
    params = KernelParams(size=8, rings=((1.0, 3.0), (3.0, 4.0)), ring_weights=(1.0, -0.5))
    state = np.ones((8, 8), dtype=np.float32) * 0.5
    a = stepper._step_numpy(state, 0.15, 0.015, 0.1, params, 0.05, 0.02, 0.0, np.random.default_rng(0))
    if not HAS_NATIVE:
        pytest.skip("native extension unavailable")
    b = stepper.step(
        state,
        mu=0.15,
        sigma=0.015,
        dt=0.1,
        kernel_params=params,
        regen_rate=0.05,
        consumption_rate=0.02,
        noise_std=0.0,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(a, b, atol=1e-4)
