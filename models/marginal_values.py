import os
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt

from gurobipy import GRB, Model


from utils_models.utils_functions import *
from utils_models.standard_model import standard_form1, construct_dual_model, standard_form


# PATHS TO THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'AJAX.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'AJAX.mps')

# results data
results_folder = os.path.join(project_root, 'figures')
marginal_values = os.path.join(results_folder, 'marginal_values')
save_path_variable = os.path.join(marginal_values, 'variables')
save_path_constraints = os.path.join(marginal_values, 'constraints')


# Create a new Gurobi model
# model = Model("OptimizationProblem")
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

def marginal_values_constraints(model, name, save_path):
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    # Get the shadow prices and corresponding constraints
    shadow_prices = model.getAttr('Pi', model.getConstrs())
    constraints = [constr.ConstrName for constr in model.getConstrs()]
    
    # Normalize the shadow prices
    max_shadow_price = max(shadow_prices, default=0)
    if max_shadow_price > 0:
        shadow_prices = [price / max_shadow_price for price in shadow_prices]
    else:
        shadow_prices = [0] * len(shadow_prices)  # Handle case where all prices are zero
    
    # Filter out zero shadow prices after normalization
    filtered_data = [(price, constr) for price, constr in zip(shadow_prices, constraints) if price != 0]
    if not filtered_data:  # If all normalized shadow prices are zero, exit function
        print(f"No non-zero normalized shadow prices for constraints in {name}.")
        return
    
    shadow_prices, constraints = zip(*filtered_data)
    
    # Plot the shadow prices
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(shadow_prices)), shadow_prices, color='skyblue')
    plt.yticks(range(len(shadow_prices)), constraints)
    plt.xlabel('Normalized Shadow Prices')
    plt.ylabel('Constraints')
    plt.title(f'Marginal Values of the Constraints {name}')
    save_in = os.path.join(save_path, f'_{name}.png')
    plt.savefig(save_in)
    plt.close()

def marginal_values_variables(model, name, save_path):
    """
    Calculate and visualize the reduced costs of the variables in a Gurobi model.
    
    Parameters:
        model (gurobipy.Model): A Gurobi model that has already been defined but must be solved in this function.
    """
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    # Get the reduced costs and corresponding variables
    reduced_costs = model.getAttr('RC', model.getVars())
    variables = [var.VarName for var in model.getVars()]
    
    # Normalize the reduced costs
    max_reduced_cost = max(reduced_costs, default=0)
    if max_reduced_cost > 0:
        reduced_costs = [cost / max_reduced_cost for cost in reduced_costs]
    else:
        reduced_costs = [0] * len(reduced_costs)  # Handle case where all costs are zero
    
    # Filter out zero reduced costs after normalization
    filtered_data = [(cost, var) for cost, var in zip(reduced_costs, variables) if cost != 0]
    if not filtered_data:  # If all normalized reduced costs are zero, exit function
        print(f"No non-zero normalized reduced costs for variables in {name}.")
        return
    
    reduced_costs, variables = zip(*filtered_data)
    
    # Plot the reduced costs
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(reduced_costs)), reduced_costs, color='skyblue')
    plt.yticks(range(len(reduced_costs)), variables)
    plt.xlabel('Normalized Reduced Costs')
    plt.ylabel('Variables')
    plt.title(f'Marginal Values of the Variables {name}')
    save_in = os.path.join(save_path, f'_{name}.png')
    plt.savefig(save_in)
    plt.close()


def marginal_values_variables(model, name, save_path):
    """
    Calculate and visualize the reduced costs of the variables in a Gurobi model.
    
    Parameters:
        model (gurobipy.Model): A Gurobi model that has already been defined but must be solved in this function.
    """
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    # Get the reduced costs and corresponding variables
    reduced_costs = model.getAttr('RC', model.getVars())
    variables = [var.VarName for var in model.getVars()]
    
    # Normalize the reduced costs
    max_reduced_cost = max(reduced_costs, default=0)
    if max_reduced_cost > 0:
        reduced_costs = [cost / max_reduced_cost for cost in reduced_costs]
    else:
        reduced_costs = [0] * len(reduced_costs)  # Handle case where all costs are zero
    
    # Filter out zero reduced costs after normalization
    filtered_data = [(cost, var) for cost, var in zip(reduced_costs, variables) if cost != 0]
    if not filtered_data:  # If all normalized reduced costs are zero, exit function
        print(f"No non-zero normalized reduced costs for variables in {name}.")
        return
    
    reduced_costs, variables = zip(*filtered_data)
    
    # Plot the reduced costs
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(reduced_costs)), reduced_costs, color='skyblue')
    plt.yticks(range(len(reduced_costs)), variables)
    plt.xlabel('Normalized Reduced Costs')
    plt.ylabel('Variables')
    plt.title(f'Marginal Values of the Variables {name}')
    save_in = os.path.join(save_path, f'_{name}.png')
    plt.savefig(save_in)
    plt.close()



if __name__ == '__main__':
    for model_name in os.listdir(GAMS_path_modified):
        if model_name.endswith('DEA.mps') or model_name.endswith('PRODSP.mps'):
            continue
        model_path = os.path.join(GAMS_path_modified, model_name)
        model = gp.read(model_path)
        name = model_path.split('/')[-1].split('.')[0]

        marginal_values_variables(model, name, save_path_variable)
        marginal_values_constraints(model, name, save_path_constraints)
