import numpy as np

from xenogenesis.substrates.ca.fitness import ca_fitness


def test_ca_fitness_penalizes_extinction():
    states = [np.zeros((4, 4), dtype=np.float32) for _ in range(3)]
    res = ca_fitness(states, mass_threshold=0.05, active_threshold=0.01)
    assert res["persistence"] == 0.0
    assert res["motility"] == 0.0
    assert res["survived"] is False


def test_ca_fitness_counts_reproduction():
    s0 = np.zeros((4, 4), dtype=np.float32)
    s0[1, 1] = 0.5
    s1 = s0.copy()
    s1[0, 0] = 0.5
    s1[3, 3] = 0.6
    res = ca_fitness([s0, s1], mass_threshold=0.02, active_threshold=0.01)
    assert res["reproduction_events"] >= 2.0
    assert res["component_count"] >= 3.0
    assert res["survived"] is True
