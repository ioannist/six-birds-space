"""Substrate generators and kernel validators."""

from .constraints import anisotropic_gate
from .grid import grid_2d
from .knn import knn_points
from .sierpinski import sierpinski
from .sphere import sphere_points
from .validate import validate_kernel

__all__ = [
    "grid_2d",
    "knn_points",
    "sphere_points",
    "sierpinski",
    "anisotropic_gate",
    "validate_kernel",
]
