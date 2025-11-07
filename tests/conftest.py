from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project modules (with spaces in the folder name) are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAZ_DIR = PROJECT_ROOT / "Gumbel Aplha Zero"
GAZ_PATH = str(GAZ_DIR)
if GAZ_PATH not in sys.path:
    sys.path.insert(0, GAZ_PATH)
