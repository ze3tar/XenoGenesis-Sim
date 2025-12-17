import importlib.util
import subprocess
import os
from pathlib import Path

import pytest


def _deps_available() -> bool:
    return all(importlib.util.find_spec(mod) is not None for mod in ("typer", "yaml", "numpy"))


def test_cli_smoke(tmp_path):
    if not _deps_available():
        pytest.skip("CLI dependencies unavailable in test environment")
    run_dir = tmp_path / "runs"
    cmd = ["python", "-m", "xenogenesis.cli", "run", "ca", "--config", "src/xenogenesis/config/defaults.yaml", "--steps", "8", "--workers", "1", "--no-render"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1], env=env)
    # ensure outputs exist
    metrics = list(Path("runs").glob("ca_*"))
    assert metrics, "run directory missing"
