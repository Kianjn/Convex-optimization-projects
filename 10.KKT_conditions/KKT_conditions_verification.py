import sympy as sp

def verify_kkt(f, g, x_star, lambda_star):
    x = list(f.free_symbols)  # Extract optimization variables
    grad_f = [sp.diff(f, var) for var in x]  # Gradient of objective function
    g_gradients = [[sp.diff(g_i, var) for var in x] for g_i in g]  # Gradients of constraints
    
    # 1. Stationarity Condition: ∇f + λ * ∇g = 0
    stationarity = [grad_f[i] + sum(lambda_star[j] * g_gradients[j][i] for j in range(len(g))) for i in range(len(x))]
    stationarity_hold = all(sp.simplify(st.subs(dict(zip(x, x_star)))).evalf() == 0 for st in stationarity)
    print(f"Stationarity holds: {stationarity_hold}")
    
    # 2. Primal Feasibility: g(x*) ≤ 0
    primal_feasibility = all(
        g_i.subs(dict(zip(x, x_star))).simplify().evalf() <= 0 if g_i.subs(dict(zip(x, x_star))).is_number else False
        for g_i in g
    )
    print(f"Primal Feasibility holds: {primal_feasibility}")
    
    # 3. Dual Feasibility: λ ≥ 0
    dual_feasibility = all(lmbda >= 0 for lmbda in lambda_star)
    print(f"Dual Feasibility holds: {dual_feasibility}")
    
    # 4. Complementary Slackness: λ_i * g_i(x*) = 0 for all i
    complementary_slackness = all(
        sp.simplify(lambda_star[i] * g[i].subs(dict(zip(x, x_star)))).evalf() == 0 for i in range(len(g))
    )
    print(f"Complementary Slackness holds: {complementary_slackness}")
    
    return stationarity_hold and primal_feasibility and dual_feasibility and complementary_slackness

# Example: Quadratic Optimization with Inequality Constraints
x1, x2 = sp.symbols('x1 x2')
f = x1**2 + x2**2  # Minimize x1² + x2²
g = [x1 + x2 - 1]  # Constraint: x1 + x2 - 1 ≤ 0

x_star = [0.5, 0.5]  # Candidate optimal solution
lambda_star = [1.0]  # Candidate Lagrange multiplier

# Verify KKT conditions
verify_kkt(f, g, x_star, lambda_star)