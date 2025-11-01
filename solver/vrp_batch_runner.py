import os
import json
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from solver.vrp_solver import solver_factory
from solver.vrp_batch_common import load_instances, validate_instance, ensure_dirs, timestamped_logger, save_stats_csv, save_stats_json

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "solver_config.yaml")

class VRPBatchRunner:
    def __init__(self, data_path, config_path: str = DEFAULT_CONFIG, max_instances: int = None):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.instances = load_instances(data_path)
        self.max_instances = max_instances or len(self.instances)
        self.outputs_dir = cfg.get("outputs_dir", "outputs")
        ensure_dirs(self.outputs_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.logger = timestamped_logger("vrp_batch", os.path.join(self.outputs_dir, "logs", f"batch_{ts}.log"))
        self.stats = []

    def run(self, solver_name: str = None):
        solver_name = solver_name or self.cfg.get("default_solver", "cbc")
        for idx, instance in enumerate(self.instances[:self.max_instances]):
            inst_id = idx + 1
            try:
                validate_instance(instance)
            except Exception as e:
                self.logger.error(f"Instance {inst_id} validation failed: {e}")
                continue
            instance_time_limit = instance.get("time_limit", self.cfg.get("default_time_limit", 30))
            solver = solver_factory(solver_name)
            self.logger.info(f"Start instance {inst_id} with solver={solver_name} time_limit={instance_time_limit}s")
            start = time.time()
            res = solver.solve(instance, time_limit=instance_time_limit)
            elapsed = time.time() - start
            res_record = {
                "instance_id": inst_id,
                "solver": solver_name,
                "objective": res.get("objective"),
                "elapsed": res.get("elapsed", elapsed),
                "used_vehicles": res.get("used_vehicles"),
                "gap": res.get("gap"),
                "best_bound": res.get("best_bound"),
                "nodes": res.get("nodes"),
                "memory": res.get("memory"),
                "status": res.get("status")
            }
            self.stats.append(res_record)
            # save per-instance figure if coords exist
            coords = instance.get("coords")
            if coords:
                fig_path = os.path.join(self.outputs_dir, "figs", f"instance_{inst_id}.png")
                try:
                    self._plot_instance_routes(coords, res, fig_path)
                except Exception as e:
                    self.logger.warning(f"Plot failed for instance {inst_id}: {e}")
            self.logger.info(f"Done instance {inst_id}: {res_record}")

        # persist stats
        csv_path = os.path.join(self.outputs_dir, "stats", "stats.csv")
        json_path = os.path.join(self.outputs_dir, "stats", "stats.json")
        save_stats_csv(self.stats, csv_path)
        save_stats_json(self.stats, json_path)
        self.logger.info(f"Saved stats to {csv_path} and {json_path}")

    def _plot_instance_routes(self, coords, res, out_path):
        # minimal plot: depot + customers. If solver produced 'vars' with x_... keys we can draw edges.
        coords = np.array(coords)
        plt.figure(figsize=(6,6))
        plt.scatter(coords[:,0], coords[:,1], c='blue')
        plt.scatter(coords[0,0], coords[0,1], c='red', marker='s', s=80, label='depot')
        vars_ = res.get("vars", {}) or {}
        # try to draw edges from vars if available
        tol = 1e-4
        for k in range(res.get("used_vehicles", 0)):
            # naive: find edges x_i_j_k == 1
            edges = []
            for key, val in vars_.items():
                if val is None:
                    continue
                if key.startswith("x_") and key.endswith(f"_{k}"):
                    parts = key.split("_")
                    try:
                        i = int(parts[1]); j = int(parts[2])
                    except:
                        continue
                    if abs(val - 1.0) < tol:
                        edges.append((i,j))
            for (i,j) in edges:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], linewidth=1)
        plt.title(f"Instance routes")
        plt.savefig(out_path, dpi=150)
        plt.close()