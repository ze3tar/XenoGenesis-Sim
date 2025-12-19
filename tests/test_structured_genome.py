import numpy as np

from xenogenesis.substrates.ca.genome import STRUCTURED_BOUNDS, StructuredGenome


def test_structured_genome_serialization_roundtrip():
    rng = np.random.default_rng(0)
    genome = StructuredGenome.random(rng)
    loaded = StructuredGenome.from_json(genome.to_json())
    assert loaded.version == genome.version
    assert loaded.distance(genome) < 1e-6


def test_structured_genome_mutation_bounds():
    rng = np.random.default_rng(1)
    genome = StructuredGenome.random(rng)
    mutated = genome.mutate(np.random.default_rng(2), sigma=0.1)

    flat_values = {
        "kernel.inner": mutated.kernel.inner,
        "kernel.outer": mutated.kernel.outer,
        "kernel.ring_ratio": mutated.kernel.ring_ratio,
        "growth.mu": mutated.growth.mu,
        "growth.sigma": mutated.growth.sigma,
        "growth.alpha": mutated.growth.alpha,
        "growth.decay": mutated.growth.decay,
        "metabolism.regen_rate": mutated.metabolism.regen_rate,
        "metabolism.consumption_rate": mutated.metabolism.consumption_rate,
        "metabolism.resource_diffusion": mutated.metabolism.resource_diffusion,
        "metabolism.biomass_diffusion": mutated.metabolism.biomass_diffusion,
        "motility.polarity_gain": mutated.motility.polarity_gain,
        "motility.polarity_decay": mutated.motility.polarity_decay,
        "motility.polarity_mobility": mutated.motility.polarity_mobility,
        "motility.polarity_noise": mutated.motility.polarity_noise,
        "division.division_threshold": mutated.division.division_threshold,
        "division.division_fraction": mutated.division.division_fraction,
        "division.elongation_trigger": mutated.division.elongation_trigger,
        "division.reproduction_cost": mutated.division.reproduction_cost,
        "environment_response.toxicity_resistance": mutated.environment_response.toxicity_resistance,
        "environment_response.resource_affinity": mutated.environment_response.resource_affinity,
        "environment_response.drift_sensitivity": mutated.environment_response.drift_sensitivity,
    }

    for key, value in flat_values.items():
        lo, hi = STRUCTURED_BOUNDS[key]
        assert lo <= value <= hi, key

    weight_sum = np.sum(np.abs(mutated.kernel.weights))
    assert np.isclose(weight_sum, 1.0)


def test_structured_genome_distance_and_mapping():
    rng = np.random.default_rng(3)
    genome = StructuredGenome.random(rng)
    params = genome.to_ca_params(grid_size=64, dt=0.05)

    assert params.grid_size == 64
    assert len(params.kernel_params.rings) == 3
    assert len(params.kernel_params.ring_weights) == 3
    assert np.isclose(sum(np.abs(params.kernel_params.ring_weights)), 1.0)

    assert genome.distance(genome) == 0.0
    mutated = genome.mutate(np.random.default_rng(4), sigma=0.02)
    assert genome.distance(mutated) > 0
