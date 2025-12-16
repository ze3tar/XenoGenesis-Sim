"""Test configuration for local imports without installing the package."""
from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure():
    """Ensure the src/ directory is importable for tests."""
    root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(root))
