import cuopt

# 创建模型
model = cuopt.Model()

# 添加变量：x, y（x为整数，y为连续变量）
x = model.add_variable(name="x", var_type="integer", lb=0, ub=10)
y = model.add_variable(name="y", var_type="continuous", lb=0, ub=10)

# 添加约束：x + 2y <= 14
model.add_constraint(x + 2 * y <= 14)

# 添加约束：3x - y >= 0
model.add_constraint(3 * x - y >= 0)

# 添加目标函数：最大化 x + y
model.set_objective(x + y, sense="max")

# 求解
solution = model.solve()

# 输出结果
print("x =", solution["x"])
print("y =", solution["y"])
print("目标值 =", solution["objective_value"])