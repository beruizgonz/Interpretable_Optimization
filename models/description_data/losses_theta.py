import os
import gurobipy as gp
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.graph_objects as go
import plotly.io as pio

from description_open_TEPES import *
from marginal_values_open_TEPES import *

# PATH TO THE DATA
parent_path = os.path.dirname(os.getcwd())
root_interpretable_optimization = os.path.dirname(parent_path)

# IMPORT FILES THAT ARE IN THE ROOT DIRECTORY
sys.path.append(parent_path)
from  models.utils_models.utils_functions import *
from models.utils_models.standard_model import *
from plot_matrix import *

# PATH TO THE DATA
real_data_path = os.path.join(parent_path, 'data/real_data')
open_tepes_9n = os.path.join(real_data_path, 'openTEPES_EAPP_2030_sc01_st1.mps')
gams_data_path = os.path.join(parent_path, 'data/GAMS_library_modified')
gams_model = os.path.join(gams_data_path, 'ROBERT.mps')

# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/marginal_values/real_problems')
results_simplified_constraints = os.path.join(results_folder, 'simplified_constraints_epsilon/importance_openTEPES_EAPP_2030_sc01_st1_constraints.json')
results_simplified_variables = os.path.join(results_folder, 'simplified_variables_percentage/openTEPES_EAPP_2030_sc01_st1_variables.json')

# Save paths 
interactive_figures = os.path.join(parent_path, 'figures_new/interactive_figures/marginal_value/simplified_constraints_percentage')
figures = os.path.join(parent_path, 'figures_new')
real_problems = os.path.join(interactive_figures, 'real_problems')
save_metrics_importance = os.path.join(figures, 'metrics_importance')
metrics_importance_real_problems = os.path.join(save_metrics_importance, 'real_problems')
save_path_metrics_pre = os.path.join(metrics_importance_real_problems, 'pre_solve')

def remove_variables_flexibility(model, values, names, percentile, metric = 'Percentage'): 
    """
    Remove variables with marginal values below a specified percentile.
    Also, determine which constraints involve these variables.
    """
    # Extract model matrices
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constrs_names = get_model_matrices(model)

    # Ensure values and names match in length
    if len(values) != len(names):
        raise ValueError(f"Length mismatch: values ({len(values)}) and names ({len(names)})")

    if metric == 'Percentage':
        abs_values = np.abs(values)
        # Sort the values in ascending order
        order = np.argsort(abs_values)
        # The way of simplifying is to delete the % with the smallest values
        variables_to_remove = []
        constrs_to_remove = []
        var_indices_to_remove = []
        for i in range(int(len(order)*percentile/100)):
            variables_to_remove.append(names[order[i]])
            var_indices_to_remove.append(order[i])
    elif metric == 'Epsilon':
        # Identify variables to remove based on threshold
        abs_values = np.abs(values)
        mask = abs_values < percentile
        variables_to_remove = np.array(names)[mask].tolist()

        # Get indices of variables to remove
        var_indices_to_remove = np.where(mask)[0]

        if len(var_indices_to_remove) == 0:
            print("No variables to remove.")
            return [], []  # Early exit if no variables meet the criteria

    # Ensure A has the correct shape
    if A.shape[1] != len(names):
        raise ValueError(f"Matrix A column size ({A.shape[1]}) does not match names length ({len(names)})")
    min_max_indices = {}  # Dictionary to track min/max indices for each row
    # Efficient constraint removal using sparse operations
    constraints_to_remove = {}
    if issparse(A):  
        A = A.tocsc()  # Convert to CSC for fast column slicing
        var_indices_to_remove = np.array(var_indices_to_remove, dtype=int)
        affected_rows = A[:, var_indices_to_remove].nonzero()[0]  
        row_indices, col_indices = A[:, var_indices_to_remove].nonzero()
        
        # Create a dictionary to track min/max for each row
        for row, col in zip(row_indices, col_indices):
            constr_name = constrs_names[row]
            actual_col = var_indices_to_remove[col]  # Map to original indices
            if row not in min_max_indices:
                min_max_indices[constr_name] = [actual_col, actual_col]  # Initialize (min, max)
            else:
                min_max_indices[constr_name][0] = min(min_max_indices[row][0], actual_col)  # Update min
                min_max_indices[constr_name][1] = max(min_max_indices[row][1], actual_col)  # Update max

    else:  # Dense matrix case
        constraints_to_remove = np.where(np.any(A[:, var_indices_to_remove] != 0, axis=1))[0]
    # Remove the variables from the model
  # Remove variables efficiently in bulk
    print(f"Removing {len(variables_to_remove)} variables from the model.")
    model.remove([model.getVarByName(var_name) for var_name in variables_to_remove])
    model.update()

    # Update constraint senses based on min/max index
    for constr_name, (min_idx, max_idx) in min_max_indices.items():
        constr = model.getConstrByName(constr_name)
        abs_min, abs_max = abs(min_idx), abs(max_idx)
        if min_idx < 0 and abs_min > abs_max:
            constr.Sense = 'G'  # Greater than or equal
        elif max_idx > 0 and abs_max > abs_min:
            constr.Sense = 'L'  # Less than or equal
        elif abs_min == abs_max:
            constr.Sense = 'L'  # Less than or equal
        else:
            constr.Sense = 'E'  # Equality

    model.update()
    return model, variables_to_remove


def flexibility_model(model, variables_to_remove, epsilon=0.015):
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constrs_names = get_model_matrices(model)

    min_max_indices = {}  # Dictionary to track min/max indices for each row
    constraints_to_remove = {}
    
    # Get variable indices
    var_indices_to_remove = {var: i for i, var in enumerate(variable_names)}
    var_indices_to_remove = [var_indices_to_remove[var] for var in variables_to_remove if var in var_indices_to_remove]

    if issparse(A):  
        A = A.tocsc()  # Convert to CSC for fast column slicing
        var_indices_to_remove = np.array(var_indices_to_remove, dtype=int)
        row_indices, col_indices = A[:, var_indices_to_remove].nonzero()

        # Track min/max for each constraint
        for row, col in zip(row_indices, col_indices):
            constr_name = constrs_names[row]
            actual_col = var_indices_to_remove[col]  # Map to original indices
            value = A[row, actual_col]  # Extract actual coefficient value

            if constr_name not in min_max_indices:
                min_max_indices[constr_name] = [value, value]  # Store (min, max values)
            else:
                min_max_indices[constr_name][0] = min(min_max_indices[constr_name][0], value)  # Update min value
                min_max_indices[constr_name][1] = max(min_max_indices[constr_name][1], value)  # Update max value

    else:  # Dense matrix case
        constraints_to_remove = np.where(np.any(A[:, var_indices_to_remove] != 0, axis=1))[0]

    # Remove variables safely
    variables_to_remove_filtered = [
        model.getVarByName(var_name) for var_name in variables_to_remove if model.getVarByName(var_name) is not None
    ]
    
    print(f"Removing {len(variables_to_remove_filtered)} variables from the model.")
    model.remove(variables_to_remove_filtered)
    model.update()

    entro1 = 0
    entro2 = 0
    entro3 = 0
    # Update constraint senses based on min/max index
    for constr_name, (min_idx, max_idx) in min_max_indices.items():
        constr = model.getConstrByName(constr_name)
        abs_min, abs_max = abs(min_idx), abs(max_idx)
        
        if abs_min < epsilon:
            entro3 += 1
            constr.Sense = 'E'  # Equality constraint
        elif min_idx < 0:
            entro1 += 1
            constr.Sense = 'G'  # Greater than or equal
        elif max_idx > 0:
            entro2 += 1
            constr.Sense = 'L'  # Less than or equal
        else:
            constr.Sense = 'E'  # Default to equality

    print(entro1, entro2, entro3)
    model.update()
    
    return model, variables_to_remove



def remove_variables(model, interest_vars):
    names_var, asos_var, group_var, invert_var = groups_by_variables(model)
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    print(A.shape)
    for i in interest_vars:
        interest_var = i
        # Get the position that start with the interest_var
        variable_map_index = {var: i for i, var in enumerate(variable_names)}
        interest_var_index = [variable_map_index[var] for var in variable_names if var.startswith(interest_var)]
        # Remove the variables from the model
        variables_rem = group_var[interest_var]
        #model.remove([model.getVarByName(var) for var in variables_rem])
        #model.update()
        # if i != 'vLineLosses':
        #     model_new, var_rem = flexibility_model(model, variables_rem)
        # else:
        #     model_new = model.copy()
        #     model_new.remove([model_new.getVarByName(var) for var in variables_rem])
        #     model_new.update()
        model_new = model.copy()
        model_new, vars = flexibility_model(model_new, variables_rem)
        model_new.setParam('OutputFlag', 0) 
        model_new.optimize()   
        A_new, _, _, _, _, _, _, _, _, _ = get_model_matrices(model_new)
        print(A_new.shape) 
        print(model_new.objVal)
        model = model_new.copy()
    return model


if __name__ == '__main__':
    model_open = gp.read(open_tepes_9n)
    model_open_s = standard_form_e2(model_open)
    remove_variables(model_open_s, ['vLineLosses','vTheta'])
