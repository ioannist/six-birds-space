"""Validate integer experiment parameters before constructing mathematical objects."""
from __future__ import annotations

import numpy as np


def config_int(value, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    return int(value)
