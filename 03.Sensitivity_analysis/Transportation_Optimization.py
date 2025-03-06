from pulp import LpMinimize, LpProblem, LpVariable, value

# Define the LP problem
model = LpProblem("Transportation_Optimization", LpMinimize)

# Decision Variables (units transported from warehouse i to store j)
x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0) for i in range(1, 4) for j in range(1, 5)}

# Cost Matrix (Transportation cost per unit from warehouse i to store j)
cost = {
    (1, 1): 4, (1, 2): 8, (1, 3): 8, (1, 4): 6,
    (2, 1): 6, (2, 2): 7, (2, 3): 3, (2, 4): 5,
    (3, 1): 9, (3, 2): 4, (3, 3): 7, (3, 4): 3
}

# Supply from each warehouse
supply = {1: 100, 2: 120, 3: 130}

# Demand at each store
demand = {1: 80, 2: 70, 3: 90, 4: 110}

# Objective function (Minimize transportation cost)
model += sum(cost[i, j] * x[i, j] for i in range(1, 4) for j in range(1, 5)), "Total_Cost"

# Supply Constraints
for i in range(1, 4):
    model += sum(x[i, j] for j in range(1, 5)) <= supply[i], f"Supply_Constraint_W{i}"

# Demand Constraints
for j in range(1, 5):
    model += sum(x[i, j] for i in range(1, 4)) == demand[j], f"Demand_Constraint_S{j}"

# Solve the LP problem
model.solve()

# Print optimal solution
print("Optimal Transportation Plan:")
for i in range(1, 4):
    for j in range(1, 5):
        print(f"Warehouse {i} to Store {j}: {value(x[i, j])} units")
print(f"Minimum Transportation Cost: ${value(model.objective)}")

# Sensitivity Analysis (Shadow Prices)
print("\nSensitivity Analysis (Shadow Prices):")
for constraint in model.constraints:
    print(f"{constraint}: Shadow Price = {model.constraints[constraint].pi}, Slack = {model.constraints[constraint].slack}")
