"""Selection utilities (NSGA-II style placeholder)."""
from __future__ import annotations
from typing import List, Dict
import numpy as np


def nsga2_select(population: List[Dict], k: int) -> List[Dict]:
    scored = sorted(population, key=lambda x: -sum(x.get("objectives", [0])))
    return scored[:k]
