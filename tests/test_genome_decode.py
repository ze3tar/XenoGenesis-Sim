import numpy as np

from xenogenesis.substrates.ca.genome import Genome, decode
from xenogenesis.substrates.ca.kernels import multi_ring_kernel


def test_decode_is_deterministic():
    genes = np.linspace(-1, 1, 24, dtype=np.float32)
    genome = Genome(genes=genes)
    params_a = decode(genome, grid_size=64, dt=0.05)
    params_b = decode(genome, grid_size=64, dt=0.05)
    assert params_a == params_b


def test_ring_ordering_and_weights_normalization():
    rng = np.random.default_rng(123)
    genome = Genome.random(24, rng)
    params = decode(genome, grid_size=64, dt=0.05)
    rings = params.kernel_params.rings
    assert rings[0][1] < rings[1][1] < rings[2][1]
    weights = np.array(params.kernel_params.ring_weights)
    assert np.isclose(np.abs(weights).sum(), 1.0, atol=1e-5)


def test_kernel_normalization():
    genes = np.zeros(24, dtype=np.float32)
    genome = Genome(genes=genes)
    params = decode(genome, grid_size=32, dt=0.05)
    kernel = multi_ring_kernel(params.kernel_params.size, params.kernel_params.rings, params.kernel_params.ring_weights)
    mass = np.abs(kernel).sum()
    assert np.isclose(mass, 1.0, atol=1e-4)
