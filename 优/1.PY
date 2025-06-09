import numpy as np
from scipy.optimize import minimize

# 原始目标函数：f(x1, x2) = - (6x1 - x1^2 - x2^2 + 2x2)
def objective(x):
    x1, x2 = x
    return x1**2 + x2**2 - 6*x1 - 2*x2

# 约束条件写成 c(x) <= 0 的形式
constraints = [
    {'type': 'ineq', 'fun': lambda x: 4 - (2*x[0] + x[1])},  # 2x1 + x2 <= 4
    {'type': 'ineq', 'fun': lambda x: 3 - (x[0] + 2*x[1])}   # x1 + 2x2 <= 3
]

# 初始点
x0 = np.array([0.5, 0.5])

# ==== 可行方向法（调用SLSQP等内点法） ====
res_feasible_dir = minimize(objective, x0, method='SLSQP', constraints=constraints)

# ==== 罚函数法 ====
def penalty_objective(x, r=1000):
    x1, x2 = x
    penalty = 0
    g1 = 2*x1 + x2 - 4
    g2 = x1 + 2*x2 - 3
    penalty += max(0, g1)**2 + max(0, g2)**2
    return objective(x) + r * penalty

res_penalty = minimize(penalty_objective, x0, method='BFGS')

# ==== 输出 ====
print("【可行方向法】结果:")
print(f"x = {res_feasible_dir.x}")
print(f"f(x) = {-res_feasible_dir.fun}")

print("\n【罚函数法】结果:")
print(f"x = {res_penalty.x}")
print(f"f(x) = {-objective(res_penalty.x)}")
