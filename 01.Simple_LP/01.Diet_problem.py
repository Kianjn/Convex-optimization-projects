from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Define the LP problem (Minimization)
diet_problem = LpProblem("Diet Optimization", LpMinimize)

# Define decision variables (amount of food in grams)
foods = {
    "Rice": {"cost": 0.025, "calories": 1.3, "protein": 0.03, "carbs": 0.28, "fat": 0.001},
    "Chicken": {"cost": 0.050, "calories": 2.5, "protein": 0.30, "carbs": 0.00, "fat": 0.04},
    "Beef": {"cost": 0.060, "calories": 2.8, "protein": 0.26, "carbs": 0.00, "fat": 0.08},
    "Lentils": {"cost": 0.030, "calories": 1.1, "protein": 0.09, "carbs": 0.20, "fat": 0.01},
    "Milk": {"cost": 0.020, "calories": 0.7, "protein": 0.03, "carbs": 0.05, "fat": 0.02},
    "Eggs": {"cost": 0.055, "calories": 1.4, "protein": 0.12, "carbs": 0.01, "fat": 0.10},
    "Vegetables": {"cost": 0.010, "calories": 0.3, "protein": 0.02, "carbs": 0.05, "fat": 0.002}
}

# Create LpVariable for each food (grams)
food_vars = {food: LpVariable(food, lowBound=0, cat='Continuous') for food in foods}

# Objective function: Minimize total cost
diet_problem += lpSum(foods[f]["cost"] * food_vars[f] for f in foods), "Total Cost"

# Constraints
diet_problem += lpSum(foods[f]["calories"] * food_vars[f] for f in foods) >= 2500, "Calorie Requirement"
diet_problem += lpSum(foods[f]["protein"] * food_vars[f] for f in foods) >= 100, "Protein Requirement"
diet_problem += lpSum(foods[f]["carbs"] * food_vars[f] for f in foods) >= 130, "Carbohydrate Requirement"
diet_problem += lpSum(foods[f]["fat"] * food_vars[f] for f in foods) <= 70, "Fat Limit"

# Solve the problem
diet_problem.solve()

# Print results
print("Optimal Diet Plan:")
for f in foods:
    print(f"{f}: {food_vars[f].varValue:.2f} grams")

print(f"\nTotal Minimum Cost: €{value(diet_problem.objective):.2f}")

#Results
# GLPSOL--GLPK LP/MIP Solver 5.0
# Parameter(s) specified in the command line:
#  --cpxlp /var/folders/4n/6ydjnj8d16x_m4by160ndjw80000gn/T/a71cda94a211434da36cb5d739aeb183-pulp.lp
#  -o /var/folders/4n/6ydjnj8d16x_m4by160ndjw80000gn/T/a71cda94a211434da36cb5d739aeb183-pulp.sol
# Reading problem data from '/var/folders/4n/6ydjnj8d16x_m4by160ndjw80000gn/T/a71cda94a211434da36cb5d739aeb183-pulp.lp'...
# 4 rows, 7 columns, 26 non-zeros
# 14 lines were read
# GLPK Simplex Optimizer 5.0
# 4 rows, 7 columns, 26 non-zeros
# Preprocessing...
# 4 rows, 7 columns, 26 non-zeros
# Scaling...
#  A: min|aij| =  1.000e-03  max|aij| =  2.800e+00  ratio =  2.800e+03
# GM: min|aij| =  1.375e-01  max|aij| =  7.274e+00  ratio =  5.292e+01
# EQ: min|aij| =  1.890e-02  max|aij| =  1.000e+00  ratio =  5.292e+01
# Constructing initial basis...
# Size of triangular part is 4
#       0: obj =   0.000000000e+00 inf =   2.882e+03 (3)
#       4: obj =   6.739891135e+01 inf =   0.000e+00 (0)
# *     6: obj =   4.841269841e+01 inf =   0.000e+00 (0)
# OPTIMAL LP SOLUTION FOUND
# Time used:   0.0 secs
# Memory used: 0.0 Mb (32525 bytes)
# Writing basic solution to '/var/folders/4n/6ydjnj8d16x_m4by160ndjw80000gn/T/a71cda94a211434da36cb5d739aeb183-pulp.sol'...

# Optimal Diet Plan:
# Rice: 1587.30 grams
# Chicken: 174.60 grams
# Beef: 0.00 grams
# Lentils: 0.00 grams
# Milk: 0.00 grams
# Eggs: 0.00 grams
# Vegetables: 0.00 grams

#Total Minimum Cost: €48.41
