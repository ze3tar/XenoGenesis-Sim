"""Metrics aggregation and output."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def save_metrics(records: list[dict], path: Path):
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
