import gurobipy as gb
import pandas as pd
import scipy as sp

from Interpretable_Optimization.models.utils_models.utils_modeling import create_original_model


# Example usage:
n_variables = 3  # Number of decision variables
n_constraints = 2  # Number of constraints

original_model = create_original_model(n_variables, n_constraints)
if original_model:
    original_model.optimize()
    print("Original Model:")
    print("Optimal Objective Value =", original_model.objVal)
    for var in original_model.getVars():
        print(f"{var.VarName} =", var.x)


