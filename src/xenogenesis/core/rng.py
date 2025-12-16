"""Central RNG helpers using PCG64DXSM."""
from numpy.random import Generator, PCG64DXSM


def make_rng(seed: int) -> Generator:
    return Generator(PCG64DXSM(seed))
