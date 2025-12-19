"""Generate rich run reports with objective summaries."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import json


def write_report(run_dir: Path):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    try:
        summary = df.describe().to_markdown()
    except ImportError:
        summary = df.describe().to_string()
    latest = df.iloc[-1].to_dict()
    objective_keys = [k for k in ("persistence", "complexity", "motility", "energy_efficiency", "entropy", "edge_density") if k in latest]
    objectives = "\n".join(f"- **{k}**: {latest[k]:.4f}" for k in objective_keys)
    report_path = run_dir / "report.md"
    species_path = run_dir / "species_summary.json"
    species_section = ""
    if species_path.exists():
        species = json.loads(species_path.read_text())
        counts = species.get("counts", {})
        species_section = "\n".join(["## Species", "", ", ".join(f"{k}: {v}" for k, v in counts.items())])
    phylo_png = run_dir / "phylogeny" / "phylogeny.png"
    phylo_section = f"![phylogeny]({phylo_png})" if phylo_png.exists() else ""
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
                "",
                species_section,
                "",
                phylo_section,
            ]
        )
    )
    return report_path
