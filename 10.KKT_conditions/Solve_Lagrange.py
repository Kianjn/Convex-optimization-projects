import numpy as np
import sympy as sp
from scipy.optimize import minimize

# Solve equality-constrained problem using Lagrange multipliers
def solve_lagrange(f, constraints):
    """Solve an equality-constrained optimization problem using Lagrange multipliers."""
    x = list(f.free_symbols)  # Extract optimization variables
    lambdas = sp.symbols(f'lambda:{len(constraints)}')  # Lagrange multipliers
    
    # Define Lagrangian function
    L = f + sum(lambdas[i] * constraints[i] for i in range(len(constraints)))
    
    # Compute KKT equations (gradients of L)
    equations = [sp.diff(L, var) for var in x] + constraints
    
    # Solve system of equations
    solution = sp.solve(equations, (*x, *lambdas), dict=True)
    return solution

# Example: minimize f(x, y) = x^2 + y^2 subject to x + y - 1 = 0
x, y = sp.symbols('x y')
f_eq = x**2 + y**2
constraints_eq = [x + y - 1]

solution_eq = solve_lagrange(f_eq, constraints_eq)
print("Solution to equality-constrained problem:", solution_eq)