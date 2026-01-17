"""
Utility functions for the pipeline.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def save_json(obj: Dict[str, Any], path: Path):
    """Save dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_npy(path: Path) -> np.ndarray:
    """Load numpy array with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)
