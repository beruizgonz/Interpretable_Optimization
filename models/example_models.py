import gurobipy as gp
from gurobipy import GRB

from utils_models.utils_functions import *
from utils_models.standard_model import standard_form_e2

# Load the data information

def toy_model():
    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVar(vtype=GRB.CONTINUOUS, lb = 0,  name="x")
    y = m.addVar(vtype=GRB.CONTINUOUS, lb = 0, name="y")
    # Set objective
    m.setObjective(x + y, GRB.MINIMIZE)
    # Add constraint: x + 2 y <= 1
    m.addConstr(-2*x + y <= 10, "c0")
    m.addConstr(0.2*x + y <= 20, "c1")
    m.addConstr(-x -y <= -5, "c2")
    # Return the model
    m.update()
    return m

def example1_model():
    model = gp.Model("Example_1")

    # Define decision variables
    x1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x1")
    x2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2")
    x3 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x3")
    x4 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x4")

    # Set the objective function: maximize z = 4x1 + 5x2 + 9x3 + 11x4
    model.setObjective(4*x1 + 5*x2 + 9*x3 + 11*x4, GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(3*x1 + 5*x2 + 10*x3 + 15*x4 <= 100, "Constraint_1")
    model.addConstr(x1 + x2 + x3 + x4 <= 15, "Constraint_2")
    model.addConstr(7*x1 + 5*x2 + 3*x3 + 2*x4 <= 120, "Constraint_3")
    model.update()
    return model

def numerical_example(): 
    model = gp.Model("Numerical_example")
    x1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x1")
    x2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2")

    model.setObjective(4/7*x1 + 3/7*x2, GRB.MAXIMIZE)

    model.addConstr(x1 + 2/7*x2 <= 2, "Constraint_1")
    model.addConstr(3/7*x1 + 5/7*x2 <= 15/7, "Constraint_2")
    model.addConstr(x1 + 6/7*x2 <= 6, "Constraint_3")
    model.addConstr(1/7*x1 + 1/7*x2<= 1, "Constraint_4")
    model.addConstr(3/7*x1 + 4/7*x2<= 12/7, "Constraint_5")

    model.update()
    return model


if __name__ == '__main__':

    model = toy_model()
    model_standard = standard_form_e2(model_gams)    
    model_standard.setParam('OutputFlag', 0)
    model_standard.optimize()
    print(model_standard.objVal)
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = calculate_bounds_candidates_sparse_improve(model_standard, None, None)
    # print the number of ub that are not inf
    ub_not_inf = [u for u in ub if u != float('inf')]
    print(len(ub_not_inf))
    print(len(ub))
    print(len(ub) - len(ub_not_inf))    
    # Create the model 
    model_bounds = build_model(A, b, c, co, lb, ub, of_sense, cons_senses, variable_names)
    model_bounds.setParam('OutputFlag', 0)  
    model_bounds.optimize()
    print(model_bounds.objVal)  