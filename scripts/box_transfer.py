"""Isolated entry point for the frozen mjlab 1.6 box-transfer policies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from unitree_box_transfer.cli import main

    main()
