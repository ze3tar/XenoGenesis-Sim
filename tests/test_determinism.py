import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pandas")
pytest.importorskip("numpy")

from xenogenesis.config import ConfigSchema
from xenogenesis.engine.sim_runner import run_ca
from pathlib import Path
import pandas as pd
import shutil

def test_deterministic_run(tmp_path):
    cfg = ConfigSchema()
    cfg.outputs.run_dir = tmp_path / "runs"
    cfg.outputs.render = False
    cfg.ca.steps = 4
    first = run_ca(cfg)
    df1 = pd.read_csv(first / "metrics.csv")
    shutil.rmtree(first)
    second = run_ca(cfg)
    df2 = pd.read_csv(second / "metrics.csv")
    assert df1.equals(df2)
