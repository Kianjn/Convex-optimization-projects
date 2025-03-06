import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Define coefficients of the objective function (maximize profit)
c = [-60, -45, -80]  # Negative because linprog does minimization by default

# Define inequality constraints (Ax <= b)
A = [
    [4, 6, 5],  # Raw Material constraint
    [10, 4, 6],  # Skilled Labor constraint
    [5, 4, 6]   # Machine Hours constraint
]
b = [120, 160, 100]  # Adjusted available resources

# Bounds (x1, x2, x3 >= 0)
x_bounds = [(0, None), (0, None), (0, None)]

# Solve the problem using the Simplex Method
result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

# Print results
if result.success:
    print("Optimal Solution:")
    print(f"x1 (Product A) = {result.x[0]:.2f}")
    print(f"x2 (Product B) = {result.x[1]:.2f}")
    print(f"x3 (Product C) = {result.x[2]:.2f}")
    print(f"Maximum Profit = {-result.fun:.2f}")
else:
    print("No optimal solution found.")

# Dual Problem Formulation
c_dual = b  # The RHS values of the primal constraints
A_dual = np.array(A).T  # Transpose of the constraint matrix
b_dual = [-ci for ci in c]  # Negative of the primal objective function coefficients

# Solve the dual problem using Simplex
result_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=[(0, None)]*3, method='highs')

# Print dual results
if result_dual.success:
    print("\nDual Solution:")
    print(f"y1 (Shadow Price of Raw Material) = {result_dual.x[0]:.2f}")
    print(f"y2 (Shadow Price of Skilled Labor) = {result_dual.x[1]:.2f}")
    print(f"y3 (Shadow Price of Machine Hours) = {result_dual.x[2]:.2f}")
    print(f"Minimum Cost (Dual Objective) = {result_dual.fun:.2f}")
else:
    print("No optimal solution found for the dual problem.")

# Geometric Method (Simplified 2-variable case: Products A and B only)
def plot_constraints():
    x = np.linspace(0, 40, 400)
    y1 = (120 - 4*x) / 6  # Raw Material constraint
    y2 = (160 - 10*x) / 4  # Skilled Labor constraint
    y3 = (100 - 5*x) / 4  # Machine Hours constraint
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label='4x1 + 6x2 ≤ 120 (Raw Material)')
    plt.plot(x, y2, label='10x1 + 4x2 ≤ 160 (Skilled Labor)')
    plt.plot(x, y3, label='5x1 + 4x2 ≤ 100 (Machine Hours)')
    
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.xlabel('Product A (x1)')
    plt.ylabel('Product B (x2)')
    plt.fill_between(x, np.minimum(y1, np.minimum(y2, y3)), 0, alpha=0.3, color='gray')
    plt.legend()
    plt.title('Feasible Region for Two-Variable Case')
    plt.grid()
    plt.show()

plot_constraints()