"""Continuous CA substrate."""
from .ca_model import CAStepper
from .fitness import ca_fitness
from .render import render_frames
from .kernels import KernelParams

__all__ = ["CAStepper", "ca_fitness", "render_frames", "KernelParams"]
