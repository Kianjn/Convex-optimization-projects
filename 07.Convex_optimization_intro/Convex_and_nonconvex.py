import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sympy import symbols, diff, sin

# 1. Convex Function Check
def is_convex(f_expr, x_sym):
    """Checks if a function is convex by computing its second derivative."""
    second_derivative = diff(diff(f_expr, x_sym), x_sym)
    return second_derivative.simplify(), second_derivative.is_nonnegative

# Define a convex function (x^2) and a non-convex function (sin(x))
x = symbols('x')
f_convex = x**2
f_nonconvex = sin(x)

second_deriv_convex, is_convex_fn = is_convex(f_convex, x)
print(f"Second derivative of f(x) = x^2: {second_deriv_convex}, Convex: {is_convex_fn}")

# 2. Convex Hull Computation
def plot_convex_hull(points):
    hull = ConvexHull(points)
    plt.scatter(points[:,0], points[:,1], label="Points")
    
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r-')
    
    plt.title("Convex Hull of Random Points")
    plt.legend()
    plt.show()

# Generate random points
np.random.seed(42)
points = np.random.rand(20, 2)
plot_convex_hull(points)

# 3. Visualization of Convex vs Non-Convex Functions
def plot_functions():
    x_vals = np.linspace(-2, 2, 100)
    y_convex = x_vals**2
    y_nonconvex = np.sin(x_vals)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_convex, label='Convex: x^2', color='blue')
    plt.title("Convex Function")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y_nonconvex, label='Non-Convex: sin(x)', color='red')
    plt.title("Non-Convex Function")
    plt.legend()
    
    plt.show()

plot_functions()
