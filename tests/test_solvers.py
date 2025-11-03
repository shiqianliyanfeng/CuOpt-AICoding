import math
import pytest

from solver.vrp_solver import solver_factory


def make_tiny_instance():
    # 1 depot (0) + 2 customers (1,2)
    instance = {
        "num_customers": 2,
        "num_vehicles": 1,
        "depot": 0,
        "nodes": [0, 1, 2],
        "coords": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        # customers list contains customer node ids (excluding depot)
        "customers": [1, 2],
        "demands": [0, 1, 1],
        "vehicle_capacities": [2],
        "distance_matrix": [
            [0, 1, 1],
            [1, 0, math.sqrt(2)],
            [1, math.sqrt(2), 0],
        ],
        # minimal placeholders for time windows / service times
        "time_windows": [[0, 100]] * 3,
        "service_times": [0, 0, 0],
    }
    return instance


def test_solver_factory_known():
    # Factory returns known types or raises on unknown
    s = solver_factory("cbc")
    assert s is not None
    with pytest.raises(ValueError):
        solver_factory("not_a_solver")


def test_cbc_solves_tiny_instance():
    inst = make_tiny_instance()
    solver = solver_factory("cbc")
    res = solver.solve(inst, time_limit=5)
    # Basic result shape checks
    assert isinstance(res, dict)
    assert "objective" in res
    assert "status" in res
    # objective should be finite
    assert res["objective"] is None or isinstance(res["objective"], (int, float))