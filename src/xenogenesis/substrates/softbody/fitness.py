"""Softbody locomotion fitness."""
from __future__ import annotations
import numpy as np
from .envs import SoftbodyEnv
from .morph_genome import VoxelMorphology
from .controller_neat import Controller


def softbody_fitness(morph: VoxelMorphology, controller: Controller) -> dict:
    env = SoftbodyEnv(morph, controller)
    pos = env.step()
    distance = float(pos[0])
    stability_penalty = 0.0
    energy = float(abs(controller.weights).sum())
    return {"distance": distance, "stability": -stability_penalty, "energy": -energy}
