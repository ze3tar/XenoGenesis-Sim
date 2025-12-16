from pathlib import Path
from xenogenesis.config import load_config
from xenogenesis.engine.sim_runner import run_softbody


def main():
    config = load_config(Path(__file__).parents[1] / "config" / "defaults.yaml")
    run_softbody(config)


if __name__ == "__main__":
    main()
