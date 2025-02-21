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
# In this code it is importance to notice that the model we are going to use is normalized. All variables are normalized to the same scale. LB = 0, UB = 1

def get_normalize_model(model_path):
    model = gp.read(model_path)  
    model_standard = standard_form_e2(model)
    model_standard = set_real_objective(model_standard)
    model_normalize = normalize_variables(model_standard)
    return model_normalize

def importance_constraints_norm(model): 
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    b_array = np.array(b)
    b_array = b_array.reshape(-1, 1)
    c_array = np.array(c)
    sum_row = np.sum(np.abs(A), axis = 1)
    imp_constr = b_array- sum_row
    return imp_constr

def name_constraint_importance(imp_constr, model): 
    constraint_names = model.getConstrs()
    constraint_names = [constr.ConstrName for constr in constraint_names]
    imp_constr = np.array(imp_constr)
    imp_constr = imp_constr.reshape(-1, 1)
    imp_constr = np.abs(imp_constr)
    imp_constr = imp_constr/np.sum(imp_constr)
    imp_constr = imp_constr.tolist()
    imp_constr = [imp_constr[i][0] for i in range(len(imp_constr))]
    imp_constr_dict = dict(zip(constraint_names, imp_constr))
    names, asos_elements, group_dict, invert_dict = groups_by_constraints(model)
    imp_constr_group = dict.fromkeys(group_dict.keys(), 0)
    for key, value in imp_constr_dict.items():
        name_group = invert_dict[key]
        imp_constr_group[name_group] += value

    # divided by the number of elements in the group
    for key, value in imp_constr_group.items():
        imp_constr_group[key] = imp_constr_group[key]/len(group_dict[key   ])
    imp_constr_group = dict(sorted(imp_constr_group.items(), key=lambda item: item[1], reverse = True))
    return imp_constr_group, group_dict

def importance_variable_of(model): 
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    c_array = np.array(c)
    sum_c = np.sum(np.abs(c_array))
    imp_var = np.abs(c_array)/sum_c
    return imp_var

def importance_variable_of_group(imp_var, model): 
    names, asos_elements, group_dict, invert_dict = groups_by_variables(model)
    A,b,c,co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    imp_var = np.array(imp_var)
    imp_var = imp_var.reshape(-1, 1)
    imp_var = imp_var.tolist()
    imp_var = [imp_var[i][0] for i in range(len(imp_var))]
    imp_var_dict = dict(zip(variable_names, imp_var))
    imp_var_group = dict.fromkeys(group_dict.keys(), 0)
    for key, value in imp_var_dict.items():
        name_group = invert_dict[key]
        imp_var_group[name_group] += value
    imp_var_group = dict(sorted(imp_var_group.items(), key=lambda item: item[1], reverse = True))
    return imp_var_group, group_dict

def importance_variable_constraints(model):
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    
    A_abs = abs(A)  # This works efficiently for sparse matrices

    # Compute row-wise sum (sum along axis 1)
    row_sums = np.array(A_abs.sum(axis=1)).flatten()  # Convert sparse sum result to 1D NumPy array

    # Avoid division by zero for rows that sum to 0
    row_sums[row_sums == 0] = 1  

    # Normalize each row
    imp_var_const = A_abs.multiply(1 / row_sums[:, np.newaxis])  # Element-wise division
    impp_var_cost = csr_matrix(imp_var_const)
    return impp_var_cost  # Returns a sparse matrix

def importance_constraint_for_variable(model):
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    imp_var_constr = importance_variable_constraints(model)
    rows_cols = np.array(imp_var_constr.sum(axis= 0)).flatten()
    imp_var_constr_norm = imp_var_constr.multiply(1/rows_cols)
    imp_var_constr_norm = csr_matrix(imp_var_constr_norm)
    print(imp_var_constr_norm.shape)
    return imp_var_constr_norm

def importance_constr_for_var_group(imp_var_constr, model):
    names_constr, asos_constr, group_constraints, invert_constr = groups_by_constraints(model)
    names_var, asos_var, group_variables, invert_var = groups_by_variables(model)
    association_dict = {(group_constr, group_var): 0 for group_constr in group_constraints for group_var in group_variables}
    pos_names = map_position_to_names(model)
    for position, name in pos_names.items():
        group_constr = invert_constr[name[0]]
        group_var = invert_var[name[1]]
        association_dict[(group_constr, group_var)] += imp_var_constr[position]
    # Normalize the values by group of constraints  
    for key, value in association_dict.items():
        group_constr = key[1]
        association_dict[key] = association_dict[key]/len(group_variables[group_constr])
    association_dict = dict(sorted(association_dict.items(), key=lambda item: item[1], reverse = True))
    return association_dict, group_constraints, group_variables


def importance_variable_constraints_group(imp_var_const, model):
    names_constr, asos_constr, group_constraints, invert_constr = groups_by_constraints(model)
    names_var, asos_var, group_variables, invert_var = groups_by_variables(model)
    association_dict = {(group_constr, group_var): 0 for group_constr in group_constraints for group_var in group_variables}
    pos_names = map_position_to_names(model)
    for position, name in pos_names.items():
        group_constr = invert_constr[name[0]]
        group_var = invert_var[name[1]]
        association_dict[(group_constr, group_var)] += imp_var_const[position]
    # Normalize the values by group of constraints  
    for key, value in association_dict.items():
        group_constr = key[0]
        association_dict[key] = association_dict[key]/len(group_constraints[group_constr])
    association_dict = dict(sorted(association_dict.items(), key=lambda item: item[1], reverse = True))
    return association_dict, group_constraints, group_variables

def synergy_variables(model): 
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    # Create a matrix with shape (n, n) where n is the number of variables
   # Get number of columns (variables)
    A_sparse = csr_matrix(A)    
    n = A_sparse.shape[1]

    # Extract the first row (efficiently for sparse matrices)
    A_first_row = A_sparse.getrow(0).toarray().flatten()  # Convert to dense array

    # Initialize synergy array (dense, will be converted back to sparse)
    synergy_variable_constraints = np.zeros((1, n))

    # Get the first element of the row
    A_00 = A_first_row[16]

    # Compute synergy values, avoiding division by zero
    for i in range(n):
        if A_first_row[i] != 0:  # Only compute if variable exists
            synergy_variable_constraints[0, i] = -A_00 / A_first_row[i]
    print(synergy_variable_constraints)

if __name__ == '__main__':
    # Load the model
    model = get_normalize_model(open_tepes_9n)
    # imp_constr = importance_constraints_norm(model)
    # name_imp = name_constraint_importance(imp_constr, model)
    # #print(name_imp) 
    # imp_var = importance_variable_of(model)
    #imp_var_group = importance_variable_of_group(imp_var, model)
    #print(imp_var_group)
    imp_var_const = importance_variable_constraints(model)
    var_constr, group_constr, group_var = importance_variable_constraints_group(imp_var_const, model)
    #print(var_constr)
    plot_group_matrix(group_var, group_constr, var_constr, 'importance_variable_constrs', 3, save_path_metrics_pre)
    # imp_constr_var = importance_constraint_for_variable(model)
    # asos_constr_var, group_constr, group_var = importance_constr_for_var_group(imp_constr_var, model)
    # plot_group_matrix(group_var, group_constr, asos_constr_var, 'importance_constraints_for_variable', 3, save_path_metrics_pre)
    
