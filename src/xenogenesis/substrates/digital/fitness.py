"""Digital organism fitness."""
from __future__ import annotations
from .instruction_vm import InstructionVM
from .genome import InstructionGenome


def digital_fitness(genome: InstructionGenome) -> dict:
    vm = InstructionVM()
    vm.reset()
    births = 0
    for op, val in genome.instructions:
        if vm.step(op, val):
            births += 1
    return {"energy": vm.energy, "births": births}
