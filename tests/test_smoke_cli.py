import subprocess
from pathlib import Path

def test_cli_smoke(tmp_path):
    run_dir = tmp_path / "runs"
    cmd = ["python", "-m", "xenogenesis.cli", "run", "ca", "--config", "src/xenogenesis/config/defaults.yaml", "--steps", "8", "--workers", "1"]
    env = {**dict(**{})}
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])
    # ensure outputs exist
    metrics = list(Path("runs").glob("ca_*"))
    assert metrics, "run directory missing"
