"""Instruction genome representation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import random


@dataclass
class InstructionGenome:
    instructions: List[Tuple[str, float]] = field(default_factory=lambda: [("push", 1.0), ("harvest", 0.0), ("replicate", 0.0)])

    def mutate(self, rate: float = 0.1):
        ops = ["push", "add", "mul", "harvest", "replicate"]
        new_instr = []
        for op, val in self.instructions:
            if random.random() < rate:
                op = random.choice(ops)
                val += random.uniform(-0.5, 0.5)
            new_instr.append((op, val))
        self.instructions = new_instr
