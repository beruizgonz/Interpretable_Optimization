from gurobipy import GRB, Model

"""
In this file, we define some simplified models to test the functionality of the optimization models.
"""

def new_model():
    model = Model("OptimizationProblem")
    
    # Add variables
    B = model.addVar(vtype=GRB.CONTINUOUS, name="B", lb=0)
    C = model.addVar(vtype=GRB.CONTINUOUS, name="C", lb=0)

    model.update()
    # Set the objective function
    model.setObjective(25 * B +20*C, GRB.MAXIMIZE)

    model.update()

    # Add constraints
    model.addConstr(20 * B + 12 * C <= 1800 , name="Constraint1")
    model.addConstr((1/15) * B + (1/15) * C <= 8 , name="Constraint2")

    model.update()
    return model

def new_model1(): 
    # Create a new model
    model = Model("LP_Optimization")

    # Define variables
    x1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x1")
    x2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x2")
    x3 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x3")

    # Set objective function
    model.setObjective(1 * x1 + 5 * x2+ 6*x3 , GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(6 * x1 + 5 * x2 + 8 * x3 <= 16, "Constraint1")
    model.addConstr(10 * x1 + 20 * x2 + 10 * x3 <= 35, "Constraint2")

    # Optimize model
    model.optimize()

    # Display the results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution:")
        print(f"x1 = {x1.x}")
        print(f"x2 = {x2.x}")
        print(f"x3 = {x3.x}")
        print(f"Objective value = {model.objVal}")
    else:
        print("No optimal solution found.")
    return model

def test_simplified_solution(model, simplified_model):
    # Create a dictionary of variable names and their corresponding values
    simplified_solution = {var.VarName: var.X for var in simplified_model.getVars()}
    # Update the original model with values from the simplified solution
    original_objective = model.getObjective()

    # Compute the objective value using the simplified solution
    obj_value = 0.0  # Initialize the objective value
    for var in model.getVars():
        if var.VarName in simplified_solution:
            # Use the value from the simplified solution
            value = simplified_solution[var.VarName]
            print(f'Using value for variable {var.VarName}: {value}')
        else:
            # Assign a default value of 0 for missing variables
            value = 0.0
            print(f'Variable {var.VarName} not in simplified solution. Using default value: {value}')
        
        # Add the contribution of this variable to the objective function
        obj_value += value * var.Obj
    print('Objective value using simplified solution:', obj_value)
    return obj_value