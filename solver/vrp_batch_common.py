import os
import json
import csv
import time
import logging
from typing import List, Dict

def ensure_dirs(base):
    paths = [
        base,
        os.path.join(base, "logs"),
        os.path.join(base, "stats"),
        os.path.join(base, "figs")
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)
    return paths

def load_instances(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # single instance file -> wrap
        return [data]
    return data

def validate_instance(instance: Dict) -> bool:
    # Basic checks: fields and sizes
    required = ["num_customers", "num_vehicles", "depot", "nodes", "customers", "distance_matrix", "demands", "vehicle_capacities"]
    for k in required:
        if k not in instance:
            raise ValueError(f"Missing required field '{k}' in instance")
    n = len(instance["nodes"])
    dm = instance["distance_matrix"]
    if len(dm) != n or any(len(row) != n for row in dm):
        raise ValueError("distance_matrix must be n x n with n == len(nodes)")
    if len(instance["demands"]) != n:
        raise ValueError("demands length must equal number of nodes")
    return True

def timestamped_logger(name: str, log_file: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def save_stats_csv(stats: List[Dict], csv_path: str):
    if not stats:
        return
    keys = list(stats[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)

def save_stats_json(stats: List[Dict], json_path: str):
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)