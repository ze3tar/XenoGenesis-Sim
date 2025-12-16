"""Minimal DEAP-style toolbox placeholder."""
from __future__ import annotations
from typing import Callable


def make_toolbox(eval_fn: Callable):
    # Placeholder to satisfy interface
    return {"evaluate": eval_fn}
