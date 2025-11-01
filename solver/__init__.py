"""
solver package exports for cuopt_vrp.

Provides convenient imports for common utilities and the solver factory:
  from solver import load_instances, ensure_dirs, solver_factory
"""
from .vrp_batch_common import (
    ensure_dirs,
    load_instances,
    validate_instance,
    timestamped_logger,
    save_stats_csv,
    save_stats_json,
)
from .vrp_solver import solver_factory, BaseSolver

__all__ = [
    "ensure_dirs",
    "load_instances",
    "validate_instance",
    "timestamped_logger",
    "save_stats_csv",
    "save_stats_json",
    "solver_factory",
    "BaseSolver",
]

__version__ = "0.1.0"