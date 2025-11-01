import os
import json
import pytest
from solver.vrp_batch_common import load_instances, validate_instance

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "vrp_dataset_100.json")

def test_load_instances():
    assert os.path.exists(DATA_FILE), "Sample dataset not found"
    insts = load_instances(DATA_FILE)
    assert isinstance(insts, list)
    assert len(insts) > 0

def test_validate_first_instance():
    insts = load_instances(DATA_FILE)
    inst = insts[0]
    # validation should not raise
    validate_instance(inst)