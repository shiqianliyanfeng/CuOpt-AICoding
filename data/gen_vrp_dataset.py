import argparse
import json
import random
from typing import List, Dict
import numpy as np


def gen_one_instance(num_customers: int = 10, num_vehicles: int = 3, seed: int | None = None) -> Dict:
    rng = np.random.RandomState(seed)
    coords = rng.rand(num_customers + 1, 2) * 100
    nodes = list(range(num_customers + 1))
    depot = 0
    customers = nodes[1:]
    distance_matrix = np.round(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2), 2).tolist()

    demands = [0] + [int(random.randint(1, 5)) for _ in customers]
    vehicle_capacities = [int(random.randint(max(10, num_customers), max(20, num_customers * 2))) for _ in range(num_vehicles)]
    vehicle_speeds = [int(random.randint(20, 60)) for _ in range(num_vehicles)]
    vehicle_fixed_costs = [int(random.randint(0, 10)) for _ in range(num_vehicles)]
    vehicle_cost_per_km = [round(random.uniform(0.005, 0.02), 4) for _ in range(num_vehicles)]

    time_windows = [(0, 100)]
    for _ in customers:
        e = random.randint(0, 50)
        l = e + random.randint(10, 60)
        time_windows.append((e, l))
    service_times = [0] + [int(random.randint(1, 10)) for _ in customers]

    beta = 10
    gamma = 20
    eta = 10
    lambda_demand = [0] + [round(50 * demands[i], 2) for i in customers]

    cost_matrix = []
    for k in range(num_vehicles):
        cost_matrix.append([[round(distance_matrix[i][j] * vehicle_cost_per_km[k], 4) for j in nodes] for i in nodes])

    return {
        "num_customers": num_customers,
        "num_vehicles": num_vehicles,
        "depot": depot,
        "nodes": nodes,
        "customers": customers,
        "coords": coords.tolist(),
        "distance_matrix": distance_matrix,
        "demands": demands,
        "vehicle_capacities": vehicle_capacities,
        "vehicle_speeds": vehicle_speeds,
        "vehicle_fixed_costs": vehicle_fixed_costs,
        "vehicle_cost_per_km": vehicle_cost_per_km,
        "time_windows": time_windows,
        "service_times": service_times,
        "beta": beta,
        "gamma": gamma,
        "eta": eta,
        "lambda_demand": lambda_demand,
        "cost_matrix": cost_matrix,
        "w_fixed": 1.0,
        "w_distance": 1.0,
        "w_early": 1.0,
        "w_late": 1.0,
        "w_unserved": 1.0,
    }


def gen_dataset(n_instances: int = 100, n_customers: int = 10, n_vehicles: int = 3, seed: int | None = None) -> List[Dict]:
    out = []
    rng = random.Random(seed)
    for i in range(n_instances):
        nc = max(1, int(n_customers + rng.randint(-2, 2)))
        nv = max(1, int(n_vehicles + rng.randint(-1, 1)))
        inst_seed = None if seed is None else seed + i
        out.append(gen_one_instance(nc, nv, seed=inst_seed))
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate VRP dataset JSON files compatible with this project")
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--n-customers", type=int, default=10)
    parser.add_argument("--n-vehicles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="./data/vrp_dataset_100.json")
    args = parser.parse_args()

    dataset = gen_dataset(args.n_instances, args.n_customers, args.n_vehicles, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(dataset)} instances to {args.out}")


if __name__ == "__main__":
    main()