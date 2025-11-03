"""
Simple entrypoint to run a demo batch solve.

Usage examples:
  python run_demo.py                               # use defaults
  python run_demo.py --data data/vrp_dataset_100.json --solver cbc --max 2
  python run_demo.py --data data/one_instance.json --solver scip --time_limit 20

This script will:
- ensure project root is on sys.path so package imports work in arbitrary environments
- create a VRPBatchRunner and run the specified solver on up to `max_instances`
- print summary and exit non-zero on unhandled errors
"""
import os
import sys
import argparse
import logging
import time

# Ensure repo root is importable when running script directly from the repo
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from solver.vrp_batch_runner import VRPBatchRunner
except Exception as e:
    # Helpful error message if package layout/imports are broken
    print("Failed to import VRPBatchRunner from solver.vrp_batch_runner:", e)
    raise

def setup_logging(level=logging.INFO):
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=level, format=fmt)

def parse_args():
    p = argparse.ArgumentParser(description="Run cuopt_vrp demo batch solver")
    p.add_argument("--data", "-d", default=os.path.join(REPO_ROOT, "data", "vrp_dataset_100.json"),
                   help="Path to instances JSON (file or list)")
    p.add_argument("--config", "-c", default=os.path.join(REPO_ROOT, "config", "solver_config.yaml"),
                   help="Path to solver config YAML")
    p.add_argument("--solver", "-s", default=None, help="Solver name override (cbc/scip/gurobi/cuopt_mip)")
    p.add_argument("--max", "-m", type=int, default=5, dest="max_instances",
                   help="Max instances to solve (default 5)")
    p.add_argument("--time_limit", "-t", type=float, default=None,
                   help="Override per-instance time limit (seconds). If not set, uses config or instance value.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    if not os.path.exists(args.data):
        logging.error("Data file not found: %s", args.data)
        sys.exit(2)
    if not os.path.exists(args.config):
        logging.warning("Config file not found, using defaults: %s", args.config)

    # Create runner
    try:
        runner = VRPBatchRunner(data_path=args.data, config_path=args.config, max_instances=args.max_instances)
    except Exception as e:
        logging.exception("Failed to construct VRPBatchRunner: %s", e)
        sys.exit(3)

    # If user provided a time limit override, inject into each instance
    if args.time_limit is not None:
        try:
            # load instances and set time_limit
            import json
            with open(args.data, "r") as f:
                insts = json.load(f)
            if isinstance(insts, dict):
                insts = [insts]
            for inst in insts:
                inst["time_limit"] = args.time_limit
            # write a temporary file for the run
            tmp_path = os.path.join(REPO_ROOT, "data", f"tmp_instances_{int(time.time())}.json")
            with open(tmp_path, "w") as f:
                json.dump(insts, f, indent=2, ensure_ascii=False)
            # re-create runner pointing to tmp file
            runner = VRPBatchRunner(data_path=tmp_path, config_path=args.config, max_instances=args.max_instances)
            logging.info("Using temporary instance file with forced time_limit=%s -> %s", args.time_limit, tmp_path)
        except Exception:
            logging.exception("Failed to set per-instance time_limit override")
            # fall back to original runner

    solver_name = args.solver or runner.cfg.get("default_solver", "cbc")
    logging.info("Starting batch run: solver=%s, data=%s, max_instances=%d", solver_name, args.data, args.max_instances)

    try:
        runner.run(solver_name=solver_name)
    except Exception as e:
        logging.exception("Batch run failed: %s", e)
        sys.exit(4)

    logging.info("Batch run finished. Stats written to: %s", os.path.join(runner.outputs_dir, "stats"))
    print("Done. See outputs/logs and outputs/stats for details.")

if __name__ == "__main__":
    main()