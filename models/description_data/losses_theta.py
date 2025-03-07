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


def flexibility_model(model, vars_remove):
    """
    Efficiently determines the minimum and maximum coefficient values for each row 
    related to the variables in `vars_remove`, and updates constraint senses accordingly.

    Parameters:
        A (scipy.sparse matrix): The constraint matrix.
        b (numpy array): The right-hand side vector of the constraints.
        cons_senses (dict): Dictionary mapping row indices to constraint senses.
        vars_remove (dict): Dictionary mapping row indices to sets of column indices to remove.

    Returns:
        cons_senses (dict): Updated constraint senses.
        new_vars (set): Set of variables affected by constraint modifications.
    """    
    new_vars = []
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    # View the rows that are affected by the variables to remove
         

    # Extract row indices where any of the specified columns have nonzero values
    row_indices, col_indices = A.nonzero()  # Get all nonzero (row, col) pairs
    affected_rows = np.unique(row_indices[np.isin(col_indices, vars_remove)])

    print('Affected_rows:', len(affected_rows))

    for row in affected_rows:
        constraint_name = constraint_names[row]  # Get the constraint name

        # Add excess variable with constraint-specific naming
        excess_var = model.addVar(
            lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, 
            name=f"excess_{constraint_name}"
        )
        new_vars.append(excess_var)
        # Add slack variable with constraint-specific naming
        slack_var = model.addVar(
            lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, 
            name=f"defect_{constraint_name}"
        )
        new_vars.append(slack_var)

    model.update()  # Ensure variables are added before modifying constraints
    # Remove the variables from the model of vars_remove
# # Retrieve all variables once (avoiding repeated list indexing)
    all_vars = model.getVars()

    # Convert vars_remove to a set for faster lookups
    vars_to_remove = [all_vars[i] for i in vars_remove]  # Set comprehension is faster

    # Remove variables in bulk
    model.remove(vars_to_remove)
    model.update()
    count = 0 
    # Modify constraints to include excess and slack variables
    for constr in model.getConstrs():
        constr_name = constr.ConstrName  # Get the constraint name
        rhs = constr.RHS  # Get the right-hand side value
        excess_var = model.getVarByName(f"excess_{constr_name}")
        slack_var = model.getVarByName(f"defect_{constr_name}")
        # Add the new constraint
        if excess_var and slack_var:
            count += 1
            row_expr = model.getRow(constr)  # Get the existing constraint expression
            
            # Modify the constraint directly
            row_expr += excess_var - slack_var  # Adjust the expression

            model.chgCoeff(constr, excess_var, 1)  # Add excess variable coefficient
            model.chgCoeff(constr, slack_var, -1) # Add the new constraint
    print('Number of constraints modified:', count)
    model.update()
    # current_obj = model.getObjective()

    # # Add the new variables with a positive sign to the existing objective
    # model.setObjective(current_obj + gp.quicksum(new_vars), GRB.MINIMIZE)
    # model.update()
    return model, vars_remove



def remove_variables(model, interest_vars):
    # Extract relevant model data
    names_var, asos_var, group_var, invert_var = groups_by_variables(model)
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)

    # Create a mapping of variable names to their indices
    variable_map_index = {var: i for i, var in enumerate(variable_names)}

    # Find all indices of variables that match ANY of the patterns in `interest_vars`
    interest_var_index = [
        i for var, i in variable_map_index.items()
        if any(var.startswith(interest) for interest in interest_vars)  # Corrected check
    ]

    # Get variable names that need to be removed
    variables_rem = [variable_names[i] for i in interest_var_index]
    print(f"Total variables to remove: {len(variables_rem)}")

    # Get upper bounds of variables to be removed
    ub_vars = [ub[i] for i in interest_var_index]
    print(f"Upper bounds of first 10 removed vars: {ub_vars[:10]}")

    # Convert variable names to their corresponding indices
    var_remove = [variable_map_index[var] for var in variables_rem if var in variable_map_index]
    print(f"Number of variables to remove: {len(var_remove)}")

    # Create a copy of the model for modifications
    model_new = model.copy()
    print(f"Original constraint matrix shape: {A.shape}")

    # Call flexibility model function to modify constraints
    model_new, vars = flexibility_model(model_new, var_remove)

    # Suppress Gurobi output
    model_new.setParam('OutputFlag', 0)
    model_new.optimize()   

    # Get new constraint matrix
    A_new, _, _, _, _, _, _, _, _, _ = get_model_matrices(model_new)
    print(f"New constraint matrix shape: {A_new.shape}") 
    print(f"Objective Value after modification: {model_new.objVal}")

    # Return the updated model
    return model_new.copy()



if __name__ == '__main__':
    model_open = gp.read(open_tepes_9n)
    model_open_s = standard_form_e2(model_open)
    A_sparse, b_sparse, c, co, lb, ub, of_sense, cons_senses, var_names = calculate_bounds_candidates_sparse(model_open_s, None, 'model_name')
    # Build model with bounds
    model_bounds = build_model(A_sparse, b_sparse, c, co, lb, ub, of_sense, cons_senses, var_names)
    model_bounds.setParam('OutputFlag', 0)
    model_bounds.optimize()
    print(model_bounds.objVal)
    model = remove_variables(model_bounds, ['vLineLosses','vTheta'])
    # names_var, asos_var, group_var, invert_var = groups_by_variables(model)
    # names_constr, asos_constr, group_constr, invert_constr = groups_by_constraints(model)
    # print(group_constr.keys())
    # postoname = map_position_to_names(model)
    # asos = create_association_dict(group_var, group_constr, invert_var, invert_constr, postoname)
    # plot_group_matrix(group_var, group_constr, asos, 'Losses_openTEPES_EAPP_2030_sc01_st1', 3, real_problems)