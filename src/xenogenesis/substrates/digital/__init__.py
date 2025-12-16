"""Digital instruction substrate."""
from .instruction_vm import InstructionVM
from .genome import InstructionGenome
from .fitness import digital_fitness

__all__ = ["InstructionVM", "InstructionGenome", "digital_fitness"]
