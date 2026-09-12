"""Pytest configuration — adds src/ to import path."""

import sys
from pathlib import Path

# Add project root/src to sys.path so tests can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
