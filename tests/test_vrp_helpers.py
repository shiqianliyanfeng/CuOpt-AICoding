import copy
from solver import vrp_batch_common


def make_valid_instance():
    inst = {
        "num_customers": 1,
        "num_vehicles": 1,
        "depot": 0,
        "nodes": [0, 1],
        "coords": [[0.0, 0.0], [1.0, 0.0]],
        "customers": [1],
        "demands": [0, 1],
        "vehicle_capacities": [2],
        "distance_matrix": [[0, 1], [1, 0]],
        "time_windows": [[0, 100], [0, 100]],
        "service_times": [0, 0],
    }
    return inst


def test_validate_accepts_valid_instance():
    inst = make_valid_instance()
    # should not raise
    vrp_batch_common.validate_instance(inst)


def test_validate_rejects_missing_field():
    inst = make_valid_instance()
    bad = copy.deepcopy(inst)
    del bad["coords"]
    try:
        vrp_batch_common.validate_instance(bad)
        assert False, "validate_instance should have raised for missing coords"
    except Exception:
        # expected
        pass
