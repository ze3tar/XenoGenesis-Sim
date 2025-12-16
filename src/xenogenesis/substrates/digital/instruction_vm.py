"""Minimal stack-based VM."""
from __future__ import annotations
from typing import List


class InstructionVM:
    def __init__(self):
        self.stack: List[float] = []
        self.energy: float = 0.0

    def reset(self):
        self.stack.clear()
        self.energy = 0.0

    def step(self, opcode: str, operand: float = 0.0):
        if opcode == "push":
            self.stack.append(operand)
        elif opcode == "add" and len(self.stack) >= 2:
            b, a = self.stack.pop(), self.stack.pop()
            self.stack.append(a + b)
        elif opcode == "mul" and len(self.stack) >= 2:
            b, a = self.stack.pop(), self.stack.pop()
            self.stack.append(a * b)
        elif opcode == "harvest" and self.stack:
            self.energy += abs(self.stack.pop())
        elif opcode == "replicate" and self.energy > 1.0:
            self.energy -= 1.0
            return True
        return False
