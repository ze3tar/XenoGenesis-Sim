"""Typer CLI for XenoGenesis-Sim."""
from __future__ import annotations
import typer
from pathlib import Path
from rich import print

from xenogenesis.config import load_config, ConfigSchema
from xenogenesis.engine.sim_runner import run_ca, run_softbody, run_digital, resume_checkpoint
from xenogenesis.analysis import plot_metrics, write_report

app = typer.Typer(help="XenoGenesis simulation CLI")


@app.command()
def run(
    substrate: str = typer.Argument(..., help="ca|softbody|ecosystem"),
    config: Path = typer.Option(Path("src/xenogenesis/config/defaults.yaml"), help="YAML config path"),
    seed: int = typer.Option(None, help="Override seed"),
    generations: int = typer.Option(None, help="Override generations"),
    pop: int = typer.Option(None, help="Override population"),
    steps: int = typer.Option(None, help="Override steps per eval"),
    workers: int = typer.Option(None, help="Worker processes"),
    render: bool = typer.Option(True, help="Render outputs (requires ffmpeg)"),
):
    cfg = load_config(config)
    if seed is not None:
        cfg.seed = seed
    if generations is not None:
        cfg.evolution.generations = generations
    if pop is not None:
        cfg.evolution.population = pop
    if steps is not None:
        cfg.ca.steps = steps
    if workers is not None:
        cfg.evolution.workers = workers
    cfg.outputs.render = render
    if substrate == "ca":
        run_ca(cfg)
    elif substrate == "softbody":
        run_softbody(cfg)
    elif substrate == "ecosystem":
        run_digital(cfg)
    else:
        raise typer.BadParameter("unknown substrate")


@app.command()
def resume(checkpoint: Path = typer.Argument(..., help="Checkpoint file")):
    state = resume_checkpoint(checkpoint)
    print(state)


@app.command()
def analyze(run: Path = typer.Option(..., help="Run directory")):
    plot_metrics(run)
    write_report(run)
    print(f"Analysis complete for {run}")


@app.command()
def doctor():
    import importlib.util

    native_ok = importlib.util.find_spec("xenogenesis_native") is not None
    ffmpeg_ok = importlib.util.find_spec("ffmpeg") is not None
    print({"native_extension": native_ok, "ffmpeg": ffmpeg_ok})


if __name__ == "__main__":
    app()
