cuopt_vrp — VRP modeling & batch solving toolkit

Overview
--------
A lightweight toolkit for modeling Vehicle Routing Problems (VRP) as MIP
and running batch experiments with multiple solvers (CBC via OR-Tools,
optional SCIP/Gurobi, and a cuOpt service client).

Getting started
---------------
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ortools
```

2. Generate a dataset (example):

```bash
python data/gen_vrp_dataset.py --n-instances 50 --n-customers 8 --out data/vrp_dataset_50.json
```

3. Run a batch solve with CBC (example):

```bash
python -m solver.vrp_batch_runner --data data/vrp_dataset_50.json --solver cbc
```

Notes
-----
- Optional solvers (SCIP/Gurobi/cuOpt) require extra installation/licensing.
- To use cuOpt service, configure the client IP/port in the solver implementation
  or pass configuration to the runner.

Support
-------
Open an issue in the repository for questions or feature requests.
