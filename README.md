# cuopt_vrp — VRP modeling & batch solving toolkit

This repository is a lightweight toolkit for modeling Vehicle Routing Problems
(VRP) as MIP and running batch experiments across multiple solvers. The code
is structured to support both local solvers (CBC via OR-Tools; optional
SCIP/Gurobi) and remote services (cuOpt routing / cuOpt MIP client).

Contents
--------
- `solver/` — solver implementations and batch runner utilities.
- `data/` — dataset generator and example datasets.
- `model/` — human-readable model documentation (English / Chinese).
- `tests/` — unit tests (self-contained, run with `pytest`).
- `run_demo.py` — a small demo entrypoint to run a short batch job.

Quick start (recommended using the project's conda env)
-----------------------------------------------------
1. Create / activate the conda environment used by this project (example):

```bash
conda create -n cuopt_env python=3.11 -y
conda activate cuopt_env
pip install -r requirements.txt
```

2. Run the unit tests:

```bash
# Use the environment python so optional deps are respected
/path/to/conda/env/bin/python -m pytest -q
# or (if your current shell has the env activated):
python -m pytest -q
```

3. Generate an example dataset:

```bash
python data/gen_vrp_dataset.py --n-instances 10 --n-customers 8 --out data/vrp_dataset_10.json
```

4. Run a short demo with the built-in batch runner (CBC):

```bash
python run_demo.py --data data/vrp_dataset_10.json --solver cbc --time-limit 10
```

Notes about solvers
-------------------
- CBC (OR-Tools) is included as a default solver and used in tests.
- SCIP and Gurobi integrations exist but require `pyscipopt` and `gurobipy` 
  respectively and are optional (tests will skip if they are not installed).
- cuOpt client code is included for remote service integration. The cuOpt
  modules require a reachable cuOpt service or the `cuopt_sh_client` Python
  package and appropriate credentials/config.

Project status & testing guidance
---------------------------------
- The test suite under `tests/` is self-contained and designed to pass using
  the default conda environment with dependencies from `requirements.txt`.
- If you do not have the optional solvers, tests will skip those cases.
- CI users: ensure tests run with an environment that either has the optional
  dependencies or rely on skips/mocks for those tests.

Further work (ideas)
--------------------
- Remove remaining dynamic import fallbacks and standardize package imports
  (recommended — reduces fragility).
- Expand test coverage for edge cases (infeasible time windows, capacity
  saturation, larger instances).
- Add example notebooks or scripts that visualize routes produced by solvers.

Contact
-------
Open issues or pull requests on the repository for questions, bug reports,
or feature requests.
cuopt_vrp — VRP modeling & batch solving toolkit
