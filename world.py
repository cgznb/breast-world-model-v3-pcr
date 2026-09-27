#!/usr/bin/env python3
"""Standalone entry point; no source path from another workflow is required."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from symm_observation.cli import main
if __name__ == "__main__":
    main()
