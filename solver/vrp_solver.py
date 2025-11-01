"""Compatibility wrapper: expose top-level vrp_solver symbols under solver.vrp_solver

Some tests and modules import `solver.vrp_solver`. The real implementation
lives at top-level `vrp_solver.py` in this repository; re-export the key
symbols here so both import styles work.
"""
from vrp_solver import solver_factory, BaseSolver

__all__ = ["solver_factory", "BaseSolver"]
