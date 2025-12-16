"""Softbody environment stub."""
from __future__ import annotations
import numpy as np
from .controller_neat import Controller
from .morph_genome import VoxelMorphology


class SoftbodyEnv:
    def __init__(self, morphology: VoxelMorphology, controller: Controller):
        self.morphology = morphology
        self.controller = controller
        self.position = np.array([0.0, 0.0])

    def step(self, steps: int = 50):
        body = self.morphology.to_array()
        for _ in range(steps):
            force = self.controller.act(body)
            self.position[0] += force * 0.01
            self.position[1] = 0.0
        return self.position.copy()
