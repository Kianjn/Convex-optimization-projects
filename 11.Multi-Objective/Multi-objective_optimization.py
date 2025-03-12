import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Objective function coefficients for cost and emissions
cost_coeffs = np.array([50, 60])  # Adjusted cost coefficients
emission_coeffs = np.array([5, 25])  # Adjusted emission coefficients

# Constraints: Ax <= b and A_eq x = b_eq
A = np.array([
    [3, 5],   # Labor constraint
    [2, 4],   # Raw material constraint
])
b = np.array([150, 120])  # Relaxed constraints

# Additional constraint: x1 + x2 >= 10 (minimum production)
A_min_production = np.array([[-1, -1]])  # Convert >= to <= for linprog
b_min_production = np.array([-10])

# Combine all constraints
A_ub = np.vstack([A, A_min_production])
b_ub = np.hstack([b, b_min_production])

# Bounds for x1 and x2
bounds = [(0, 30), (0, 20)]

# Weights for the Weighted Sum Method
weights = np.linspace(0, 1, 20)

cost_values = []
emission_values = []
solutions = []

# Solve for different weights
for w in weights:
    # Combine objectives: Minimize cost and emissions
    obj_coeffs = w * cost_coeffs - (1 - w) * emission_coeffs
    res = linprog(c=obj_coeffs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        x1_opt, x2_opt = res.x
        total_cost = np.dot(cost_coeffs, res.x)
        total_emission = np.dot(emission_coeffs, res.x)
        
        cost_values.append(total_cost)
        emission_values.append(total_emission)
        solutions.append((x1_opt, x2_opt))
    else:
        cost_values.append(None)
        emission_values.append(None)
        solutions.append((None, None))

# Plot Pareto Front
plt.figure(figsize=(8, 6))
valid_points = [(e, c) for e, c in zip(emission_values, cost_values) if e is not None and c is not None]
if valid_points:
    emissions, costs = zip(*valid_points)
    plt.plot(emissions, costs, 'bo-', label='Pareto Front')
plt.xlabel('Total Carbon Emissions (kg CO₂)')
plt.ylabel('Total Cost ($)')
plt.title('Pareto Front: Cost vs. Emissions')
plt.legend()
plt.grid()
plt.show()

# Display results
for i, (cost, emission, sol) in enumerate(zip(cost_values, emission_values, solutions)):
    if sol[0] is not None:
        print(f"w={weights[i]:.2f}: x1={sol[0]:.2f}, x2={sol[1]:.2f}, Cost={cost:.2f}, Emissions={emission:.2f}")
    else:
        print(f"w={weights[i]:.2f}: No feasible solution")