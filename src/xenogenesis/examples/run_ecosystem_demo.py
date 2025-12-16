from pathlib import Path
from xenogenesis.config import load_config
from xenogenesis.engine.sim_runner import run_digital


def main():
    config = load_config(Path(__file__).parents[1] / "config" / "defaults.yaml")
    run_digital(config)


if __name__ == "__main__":
    main()
