from gurobipy import Model, GRB
import pulp
import cvxpy as cp

# ---- SOLVING WITH GUROBI ---- #
model = Model("Diet Optimization")
x_A = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_A")
x_B = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_B")

model.setObjective(3*x_A + 2*x_B, GRB.MINIMIZE)
model.addConstr(250*x_A + 200*x_B >= 500, "Calorie Constraint")
model.addConstr(10*x_A + 5*x_B >= 20, "Protein Constraint")

model.optimize()
print("\nGurobi Solution:")
for v in model.getVars():
    print(f"{v.varName} = {v.x}")
print(f"Optimal Cost = {model.objVal}")

# ---- SOLVING WITH PuLP ---- #
prob = pulp.LpProblem("Diet_Optimization", pulp.LpMinimize)
x_A_pulp = pulp.LpVariable("x_A", lowBound=0, cat='Continuous')
x_B_pulp = pulp.LpVariable("x_B", lowBound=0, cat='Continuous')

prob += 3*x_A_pulp + 2*x_B_pulp
prob += 250*x_A_pulp + 200*x_B_pulp >= 500
prob += 10*x_A_pulp + 5*x_B_pulp >= 20

prob.solve()
print("\nPuLP Solution:")
for v in prob.variables():
    print(f"{v.name} = {v.varValue}")
print(f"Optimal Cost = {pulp.value(prob.objective)}")

# ---- SOLVING WITH CVXPY ---- #
x_A_cvx = cp.Variable(nonneg=True)
x_B_cvx = cp.Variable(nonneg=True)

objective = cp.Minimize(3*x_A_cvx + 2*x_B_cvx)
constraints = [
    250*x_A_cvx + 200*x_B_cvx >= 500,
    10*x_A_cvx + 5*x_B_cvx >= 20
]

prob_cvx = cp.Problem(objective, constraints)
prob_cvx.solve()
print("\nCVXPY Solution:")
print(f"x_A = {x_A_cvx.value}")
print(f"x_B = {x_B_cvx.value}")
print(f"Optimal Cost = {prob_cvx.value}")
