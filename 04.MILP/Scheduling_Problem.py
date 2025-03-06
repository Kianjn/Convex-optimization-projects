from gurobipy import Model, GRB

# Given Data
jobs = [1, 2, 3]
processing_time = {1: 3, 2: 2, 3: 4}
deadline = {1: 8, 2: 6, 3: 10}
weight = {1: 2, 2: 1, 3: 3}
M = 100  # Large constant

# Create Model
model = Model("Integer_Scheduling")

# Decision Variables
S = model.addVars(jobs, vtype=GRB.CONTINUOUS, name="Start")
C = model.addVars(jobs, vtype=GRB.CONTINUOUS, name="Completion")
x = model.addVars(jobs, jobs, vtype=GRB.BINARY, name="Order")

# Constraints
for i in jobs:
    model.addConstr(C[i] == S[i] + processing_time[i], name=f"Completion_{i}")
    model.addConstr(C[i] <= deadline[i], name=f"Deadline_{i}")
    
    for j in jobs:
        if i != j:
            model.addConstr(S[i] + processing_time[i] <= S[j] + M * (1 - x[i, j]),
                            name=f"Order_{i}_{j}")
            model.addConstr(S[j] + processing_time[j] <= S[i] + M * x[i, j],
                            name=f"Order_{j}_{i}")

# Objective Function (Minimize weighted completion time)
model.setObjective(sum(weight[j] * C[j] for j in jobs), GRB.MINIMIZE)

# Solve Model
model.optimize()

# Print Solution
if model.status == GRB.OPTIMAL:
    print("Optimal Schedule:")
    for j in jobs:
        print(f"Job {j}: Start at {S[j].x}, Complete at {C[j].x}")
    print(f"Total Weighted Completion Time: {model.objVal}")
