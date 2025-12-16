from pathlib import Path
from xenogenesis.engine.checkpointing import save_checkpoint, load_checkpoint

def test_checkpoint_roundtrip(tmp_path):
    state = {"gen": 5, "best": [1, 2, 3]}
    path = tmp_path / "chk.db"
    save_checkpoint(path, state)
    loaded = load_checkpoint(path)
    assert loaded == state
