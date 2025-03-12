import cvxpy as cp
import numpy as np

# Define the symmetric matrix variable X (2x2)
X = cp.Variable((2, 2), symmetric=True)

# Define the scalar variable lambda (smallest eigenvalue to maximize)
lambda_min = cp.Variable()

# Identity matrix of size 2x2
I = np.eye(2)

# Constraints
constraints = [
    X - lambda_min * I >> 0,  # Ensures lambda_min is the smallest eigenvalue
    X >> 0,  # Ensures X is PSD
    cp.trace(X) == 1  # Prevents unbounded growth
]

# Define the objective function (maximize lambda_min)
objective = cp.Maximize(lambda_min)

# Define the problem
problem = cp.Problem(objective, constraints)

print("Solving SDP problem...")
problem.solve(solver=cp.CVXOPT)
print("Solver status:", problem.status)
print("Optimal minimum eigenvalue:", lambda_min.value)
print("Optimal X matrix:\n", X.value)