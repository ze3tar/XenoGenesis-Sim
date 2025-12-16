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
    fig, ax = plt.subplots()
    for col in df.columns:
        if col == "step":
            continue
        ax.plot(df.get("step", range(len(df))), df[col], label=col)
    ax.set_xlabel("step")
    ax.legend()
    out = run_dir / "plots" / "fitness.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
