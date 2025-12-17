"""Generate rich run reports with objective summaries."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_report(run_dir: Path):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    summary = df.describe().to_markdown()
    latest = df.iloc[-1].to_dict()
    objective_keys = [k for k in ("persistence", "complexity", "motility", "energy_efficiency", "entropy", "edge_density") if k in latest]
    objectives = "\n".join(f"- **{k}**: {latest[k]:.4f}" for k in objective_keys)
    report_path = run_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Run Report",
                "",
                "## Objectives (final snapshot)",
                objectives,
                "",
                "## Metrics summary",
                summary,
            ]
        )
    )
    return report_path
