# cuopt_vrp — 车辆路径问题（VRP）建模与批量求解工具箱

本仓库提供一个轻量的工具集合，用于将车辆路径问题（VRP）建模为整数规划（MIP），并在多种求解器上进行批量试验。代码支持本地求解器（如 OR-Tools/CBC；可选的 SCIP/Gurobi）和远程服务（如 cuOpt 路由/CSR-MIP 客户端）。

目录结构
---------
- `solver/` — 求解器实现与批量运行工具。
- `data/` — 数据生成脚本与示例数据集。
- `model/` — 模型文档（中/英双语）。
- `tests/` — 单元测试（自包含，可用 `pytest` 运行）。
- `run_demo.py` — 一个用于快速演示的小脚本。

快速开始（建议使用 conda 环境）
---------------------------------
1. 创建并激活 conda 环境（示例）：

```bash
conda create -n cuopt_env python=3.11 -y
conda activate cuopt_env
pip install -r requirements.txt
```

2. 运行单元测试：

```bash
# 使用环境的 python 来运行测试，以确保可选依赖得到尊重
/path/to/conda/env/bin/python -m pytest -q
# 或（在已激活环境中）：
python -m pytest -q
```

3. 生成示例数据集：

```bash
python data/gen_vrp_dataset.py --n-instances 10 --n-customers 8 --out data/vrp_dataset_10.json
```

4. 使用 CBC 运行一个简短演示：

```bash
python run_demo.py --data data/vrp_dataset_10.json --solver cbc --time-limit 10
```

关于求解器的说明
-----------------
- 默认使用 CBC（通过 OR-Tools）作为本地求解器，测试也以此为基线。
- SCIP / Gurobi 的接口已包含，但需要分别安装 `pyscipopt` / `gurobipy`。
- cuOpt 客户端模块用于与远程 cuOpt 服务交互。使用这些模块需要可用的
  cuOpt 服务或安装 `cuopt_sh_client` 并配置凭据。

项目与测试建议
-----------------
- `tests/` 下的测试用例已尽量自包含，可在没有全部可选依赖的环境下
  通过（通过 skip 实现）。
- CI 场景下，请确保为可选依赖准备好替代方案，或在有能力的 runner 上
  安装这些依赖。

后续改进（建议）
-----------------
- 移除代码中的动态导入与 sys.path 注入，统一使用包内导入（推荐）。
- 扩展测试覆盖率，加入更多边界情况（不可行时间窗、容量极限、大规模实例等）。
- 添加用于可视化路线的示例 notebook 或脚本。

联系方式
--------
如有问题、bug 或功能请求，请在仓库中打开 Issue 或提交 PR。
cuopt_vrp — 车辆路径问题（VRP）建模与批量求解工具箱