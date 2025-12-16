"""Soft-body substrate stubs."""
from .morph_genome import VoxelMorphology
from .controller_neat import Controller
from .envs import SoftbodyEnv
from .fitness import softbody_fitness

__all__ = ["VoxelMorphology", "Controller", "SoftbodyEnv", "softbody_fitness"]
