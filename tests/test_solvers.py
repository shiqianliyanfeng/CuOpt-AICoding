import pytest
import os
from vrp_solver import solver_factory

def make_trivial_instance():
    # depot 0, one customer 1, single vehicle
    nodes = [0, 1]
    return {
        "num_customers": 1,
        "num_vehicles": 1,
        "depot": 0,
        "nodes": nodes,
        "customers": [1],
        "distance_matrix": [[0.0, 1.0], [1.0, 0.0]],
        "demands": [0, 1],
        "vehicle_capacities": [10],
        "vehicle_fixed_costs": [0],
        "vehicle_cost_per_km": [1.0],
        "service_times": [0, 0],
        "vehicle_speeds": [1],
        "time_windows": [(0, 100), (0, 100)],
        "w_fixed": 1.0,
        "w_distance": 1.0,
        "w_early": 1.0,
        "w_late": 1.0,
        "w_unserved": 1.0,
    }


def test_solver_factory_returns_known_solvers():
    # CBC should be available
    s = solver_factory('cbc')
    assert s is not None


def test_cbc_solver_trivial_instance():
    inst = make_trivial_instance()
    s = solver_factory('cbc')
    res = s.solve(inst, time_limit=2)
    assert isinstance(res, dict)
    assert 'objective' in res and 'elapsed' in res and 'status' in res


def test_cuopt_mip_availability():
    # If cuopt_sh_client is installed, the solver should instantiate; otherwise skip
    try:
        solver = solver_factory('cuopt')
    except Exception as e:
        pytest.skip(f"cuOpt client not available: {e}")
    # if we get here, solver is available; solve should return a dict (may contact service)
    inst = make_trivial_instance()
    res = solver.solve(inst, time_limit=1)
    assert isinstance(res, dict)
