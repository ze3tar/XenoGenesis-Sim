import pytest

pytest.importorskip("numpy")
import numpy as np

from xenogenesis.substrates.ca.ca_model import CAStepper, HAS_NATIVE


def test_ca_step_equivalence():
    stepper = CAStepper()
    state = np.ones((8, 8), dtype=np.float32) * 0.5
    a = stepper._step_numpy(state, 0.15, 0.015, 0.1, 2.0, 3.0, 0.5)
    if not HAS_NATIVE:
        pytest.skip("native extension unavailable")
    b = stepper.step(state, mu=0.15, sigma=0.015, dt=0.1, inner_radius=2.0, outer_radius=3.0, ring_ratio=0.5)
    assert np.allclose(a, b, atol=1e-4)
