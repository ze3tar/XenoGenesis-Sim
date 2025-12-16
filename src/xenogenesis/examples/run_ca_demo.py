from xenogenesis.config import load_config
from xenogenesis.engine.sim_runner import run_ca
from pathlib import Path


def main():
    config = load_config(Path(__file__).parents[1] / "config" / "defaults.yaml")
    run_ca(config)


if __name__ == "__main__":
    main()
