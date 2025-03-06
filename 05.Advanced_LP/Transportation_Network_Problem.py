import gurobipy as gp
from gurobipy import GRB

# Define Nodes, Arcs, and Modes
nodes = ['Factory1', 'Factory2', 'Port1', 'Port2', 'City1', 'City2', 'City3', 'Sink']
modes = ['Truck', 'Train', 'Subway']

# Supply and Demand Values
supply = {'Factory1': 100, 'Factory2': 150, 'Port1': 200, 'Port2': 250, 'City1': 0, 'City2': 0, 'City3': 0, 'Sink': 0}
demand = {'Factory1': 0, 'Factory2': 0, 'Port1': 0, 'Port2': 0, 'City1': 150, 'City2': 200, 'City3': 300, 'Sink': 50}  # Sink absorbs extra supply

# Transportation Costs (per unit flow)
arcs = {
    ('Factory1', 'Port1', 'Truck'): 10,
    ('Factory1', 'City1', 'Train'): 15,
    ('Factory2', 'Port2', 'Truck'): 12,
    ('Factory2', 'City2', 'Train'): 18,
    ('Port1', 'City1', 'Truck'): 20,
    ('Port1', 'City2', 'Train'): 22,
    ('Port2', 'City2', 'Truck'): 25,
    ('Port2', 'City3', 'Train'): 28,
    ('City1', 'City2', 'Subway'): 5,
    ('City2', 'City3', 'Subway'): 7,
    ('City3', 'Sink', 'Truck'): 1  # Sink connection
}

# Capacity Limits
capacity = {arc: 999 for arc in arcs}  # High capacity to prevent artificial constraints

# Initialize Model
model = gp.Model("MultiModalTransport")

# Define Decision Variables
flow = {arc: model.addVar(lb=0, ub=capacity[arc], vtype=GRB.CONTINUOUS, name=f"flow_{arc[0]}_{arc[1]}_{arc[2]}") for arc in arcs}

# Define Objective Function (Minimize Cost + Congestion)
model.setObjective(gp.quicksum(arcs[arc] * flow[arc] + 0.1 * flow[arc] * flow[arc] for arc in arcs), GRB.MINIMIZE)

# Flow Conservation Constraints
for node in nodes:
    inflow = gp.quicksum(flow[arc] for arc in arcs if arc[1] == node)
    outflow = gp.quicksum(flow[arc] for arc in arcs if arc[0] == node)
    model.addConstr(inflow - outflow == demand[node] - supply[node], name=f"balance_{node}")

# Optimize Model
model.optimize()

# Print Results
if model.status == GRB.OPTIMAL:
    for arc in arcs:
        if flow[arc].x > 0:
            print(f"Flow from {arc[0]} to {arc[1]} via {arc[2]}: {flow[arc].x}")
    print("Total Cost:", model.objVal)
