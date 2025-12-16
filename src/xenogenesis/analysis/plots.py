"""Plotting helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(run_dir: Path):
    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.exists():
        return None
    df = pd.read_parquet(metrics_path)
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for col in df.columns:
        if col == "step" or col == "descriptor":
            continue
        ax.plot(df.get("step", range(len(df))), df[col], label=col)
    ax.set_xlabel("step")
    ax.legend()
    out = out_dir / "fitness.png"
    fig.savefig(out)
    plt.close(fig)

    if {"persistence", "complexity"}.issubset(df.columns):
        fig2, ax2 = plt.subplots()
        ax2.scatter(df["persistence"], df["complexity"], c=df.get("motility", None), cmap="viridis")
        ax2.set_xlabel("persistence")
        ax2.set_ylabel("complexity")
        ax2.set_title("Behavior cloud")
        fig2.colorbar(ax2.collections[0], label="motility")
        pareto_path = out_dir / "pareto.png"
        fig2.savefig(pareto_path)
        plt.close(fig2)
    return out
