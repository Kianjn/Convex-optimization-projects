import cvxpy as cp
import numpy as np

# Define problem data
n = 10  # Number of assets
stocks = ["Apple", "Nvidia", "Google", "Tesla", "Amazon", "Microsoft", "Meta", "Berkshire Hathaway", "Johnson & Johnson", "JPMorgan"]

# Expected returns for each stock
r = np.array([0.15, 0.20, 0.18, 0.22, 0.17, 0.16, 0.19, 0.13, 0.12, 0.14])
R_t = 0.16  # Target return

# Covariance matrix (risk) for the 10 stocks
Sigma = np.array([
    [0.10, 0.02, 0.04, 0.01, 0.03, 0.02, 0.05, 0.01, 0.00, 0.02],
    [0.02, 0.12, 0.06, 0.02, 0.04, 0.03, 0.05, 0.02, 0.01, 0.03],
    [0.04, 0.06, 0.15, 0.03, 0.06, 0.05, 0.07, 0.03, 0.02, 0.04],
    [0.01, 0.02, 0.03, 0.20, 0.05, 0.04, 0.06, 0.02, 0.01, 0.03],
    [0.03, 0.04, 0.06, 0.05, 0.18, 0.06, 0.08, 0.04, 0.02, 0.05],
    [0.02, 0.03, 0.05, 0.04, 0.06, 0.14, 0.07, 0.03, 0.02, 0.04],
    [0.05, 0.05, 0.07, 0.06, 0.08, 0.07, 0.22, 0.05, 0.03, 0.06],
    [0.01, 0.02, 0.03, 0.02, 0.04, 0.03, 0.05, 0.11, 0.02, 0.03],
    [0.00, 0.01, 0.02, 0.01, 0.02, 0.02, 0.03, 0.02, 0.09, 0.02],
    [0.02, 0.03, 0.04, 0.03, 0.05, 0.04, 0.06, 0.03, 0.02, 0.13]
])

# Define decision variable
x = cp.Variable(n)

# Define objective function (minimize risk)
objective = cp.Minimize(0.5 * cp.quad_form(x, Sigma))

# Constraints
constraints = [
    cp.sum(x) == 1,        # Budget constraint
    r @ x >= R_t,          # Expected return constraint
    x >= 0                 # No short-selling constraint
]

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Display results
print("Optimal Portfolio Allocation:")
for i in range(n):
    print(f"{stocks[i]}: {x.value[i]:.4f}")
print("Minimum Portfolio Risk (Variance):", problem.value)
