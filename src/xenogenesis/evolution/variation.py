"""Mutation helpers."""
from __future__ import annotations
from typing import List, Dict
import copy
import numpy as np


def mutate_population(pop: List[Dict], rate: float) -> List[Dict]:
    rng = np.random.default_rng()
    new_pop = []
    for ind in pop:
        clone = copy.deepcopy(ind)
        if rng.random() < rate and "genome" in clone:
            if hasattr(clone["genome"], "mutate"):
                clone["genome"].mutate(rate)
        new_pop.append(clone)
    return new_pop
