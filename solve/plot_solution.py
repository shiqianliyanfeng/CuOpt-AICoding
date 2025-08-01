import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_routes(model_data, solution, tol=1e-4):
        """可视化车辆路径，tol为浮点误差容忍"""
        coords = np.array(model_data["coords"])
        sol = solution["solver_response"]["solution"]
        vars = sol.get("vars", {})
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', label='demand')
        plt.scatter(coords[0, 0], coords[0, 1], c='red', label='depot', marker='s')
        for k in range(model_data["num_vehicles"]):
            if vars.get(f"y_{k}", 0) < 0.5:
                continue
            # 构建路径
            route = [model_data["depot"]]
            current = model_data["depot"]
            visited = set([current])
            while True:
                found = False
                for j in model_data["nodes"]:
                    if j != current and abs(vars.get(f"x_{current}_{j}_{k}", 0.0) - 1.0) < tol and j not in visited:
                        plt.plot([coords[current, 0], coords[j, 0]], [coords[current, 1], coords[j, 1]], label=f'Vehicle {k}' if len(route)==1 else "", linewidth=2)
                        route.append(j)
                        visited.add(j)
                        current = j
                        found = True
                        break
                if not found:
                    # 回到depot
                    if abs(vars.get(f"x_{current}_{model_data['depot']}_{k}", 0.0) - 1.0) < tol:
                        plt.plot([coords[current, 0], coords[model_data['depot'], 0]], [coords[current, 1], coords[model_data['depot'], 1]], color='gray', linestyle='--')
                    break
        plt.legend()
        plt.title("VRP Solution Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

#current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()
with open('./solve/model_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
with open('./solve/solution_plot.json', 'r', encoding='utf-8') as f:
        solution = json.load(f)
plot_routes(data, solution)