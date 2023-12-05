import gurobipy as gb
import pandas as pd
import scipy as sp
import os

from Interpretable_Optimization.models.utils_models.utils_modeling import create_original_model

# Get the directory of the current script (main.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up one levels to get to the root of the project
project_root = os.path.dirname(current_script_dir)

data_path = os.path.join(project_root, 'data')
model_filename = os.path.join(data_path, "original_model.mps")

# creating the original_model
n_variables = 3  # Number of decision variables
n_constraints = 2  # Number of constraints
original_model = create_original_model(n_variables, n_constraints)
if original_model:
    original_model.optimize()
    print("Original Model:")
    print("Optimal Objective Value =", original_model.objVal)
    for var in original_model.getVars():
        print(f"{var.VarName} =", var.x)

# saving the original model
original_model.write(model_filename)

# creating the presolved model
presolved_model = original_model.presolve()

# Access constraint matrix A and right-hand side vector b of the original model
A = original_model.getA()
b = original_model.getAttr('RHS', original_model.getConstrs())

