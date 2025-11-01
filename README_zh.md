cuopt_vrp — 车辆路径问题（VRP）建模与批量求解工具箱

概述
----
一个轻量级工具，用于将 VRP 建模为 MIP 并使用多种求解器进行批量实验（支持 OR-Tools/CBC，可选 SCIP/Gurobi 和 cuOpt 服务）。

快速开始
--------
1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ortools
```

2. 生成数据示例：

```bash
python data/gen_vrp_dataset.py --n-instances 50 --n-customers 8 --out data/vrp_dataset_50.json
```

3. 使用 CBC 批量求解示例：

```bash
python -m solver.vrp_batch_runner --data data/vrp_dataset_50.json --solver cbc
```

说明
---
- 可选求解器（SCIP/Gurobi/cuOpt）通常需要额外安装或许可证。
- 若使用 cuOpt 服务，请在求解器实现中配置 IP/端口，或在运行器中传递配置。

支持
---
在仓库中提交 issue 以提问或请求新功能。
