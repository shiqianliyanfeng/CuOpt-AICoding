import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Solver interface
class BaseSolver(ABC):
    @abstractmethod
    def solve(self, instance: Dict, time_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Solve one instance.

        Returns a dict with keys (at minimum):
          - objective (float or None)
          - elapsed (seconds)
          - used_vehicles (int or None)
          - gap, best_bound, nodes, memory (may be None)
          - status (str)
          - vars (dict) optional raw variables
        """
        pass

# Factory
def solver_factory(name: str):
    name = name.lower()
    if name in ("cbc", "ortools", "or-tools"):
        from solver.solver_impl_cbc import SolverCBC
        return SolverCBC()
    if name == "scip":
        from solver.solver_impl_scip import SolverSCIP
        return SolverSCIP()
    if name == "gurobi":
        from solver.solver_impl_gurobi import SolverGurobi
        return SolverGurobi()
    if name in ("cuopt", "cuopt_sh"):
        from solver.solver_impl_cuopt_vrp import SolverCuOpt
        return SolverCuOpt()
    if name in ("cuopt_mip", "cuopt-mip"):
        from solver.solver_impl_cuopt_mip import SolverCuOptMIP
        return SolverCuOptMIP()
    raise ValueError(f"Unknown solver '{name}'")