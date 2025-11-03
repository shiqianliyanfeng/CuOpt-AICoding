"""Package-local VRP solver interface and factory.

This module provides a small `BaseSolver` abstract base class and a
`solver_factory(name)` that returns a solver instance by importing the
implementation from this package's modules. This avoids relying on a
top-level `vrp_solver.py` file and keeps imports local to the package.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseSolver(ABC):
	@abstractmethod
	def solve(self, instance: Dict, time_limit: Optional[float] = None) -> Dict[str, Any]:
		"""Solve one instance and return a standardized result dict."""


def solver_factory(name: str):
	"""Return a solver instance by name.

	Supported names: 'cbc', 'scip', 'gurobi', 'cuopt', 'cuopt_mip'
	"""
	name = (name or "cbc").lower()
	if name in ("cbc", "ortools", "or-tools"):
		from .solver_impl_cbc import SolverCBC

		return SolverCBC()
	if name == "scip":
		from .solver_impl_scip import SolverSCIP

		return SolverSCIP()
	if name == "gurobi":
		from .solver_impl_gurobi import SolverGurobi

		return SolverGurobi()
	if name in ("cuopt", "cuopt_sh"):
		from .solver_impl_cuopt_vrp import SolverCuOptVRP

		return SolverCuOptVRP()
	if name in ("cuopt_mip", "cuopt-mip"):
		from .solver_impl_cuopt_mip import SolverCuOptMIP

		return SolverCuOptMIP()
	raise ValueError(f"Unknown solver '{name}'")


__all__ = ["BaseSolver", "solver_factory"]
