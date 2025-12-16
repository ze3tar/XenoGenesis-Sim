# AGENTS.md for XenoGenesis-Sim

This file provides **structured instructions for coding agents** (e.g., GitHub Copilot coding agent, CLI-based AI tools, etc.) to work effectively on the XenoGenesis-Sim repository.  
This document supplements `README.md` by giving agents actionable commands, boundaries, technical context, and conventions tailored to this project. :contentReference[oaicite:1]{index=1}

---

## Project Overview

XenoGenesis-Sim is a **Linux-compatible artificial life and evolutionary simulation platform** that models digital organisms under arbitrary planetary conditions.  
Agents should understand that the goal is to:
- Build, test, and evolve simulation modules
- Maintain modular evolutionary substrates (e.g., digital genomes, continuous cellular automata, soft-body physics, neural evolution)
- Ensure reproducibility, documentation, and rigorous test coverage

---

## Environment Setup

### Required Tools
Agents should assume the following tools are installed and used:
```bash
# Base development tools
sudo apt update && sudo apt install -y build-essential cmake git python3 python3-pip

# Python and dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Example Python tooling
pip install pytest black flake8
