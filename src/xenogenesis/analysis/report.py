"""Generate run report."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_report(run_dir: Path):
    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.exists():
        return None
    df = pd.read_parquet(metrics_path)
    summary = df.describe().to_markdown()
    report_path = run_dir / "report.md"
    report_path.write_text(f"# Run Report\n\n{summary}\n")
    return report_path
