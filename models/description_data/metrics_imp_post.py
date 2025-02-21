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
from metrics_importance import get_normalize_model

# PATH TO THE DATA
parent_path = os.path.dirname(os.getcwd())
root_interpretable_optimization = os.path.dirname(parent_path)

# IMPORT FILES THAT ARE IN THE ROOT DIRECTORY
sys.path.append(parent_path)
from  models.utils_models.utils_functions import *
from models.utils_models.standard_model import *
from models.utils_models.canonical_model import *
from plot_matrix import *

# PATH TO THE DATA
real_data_path = os.path.join(parent_path, 'data/real_data')
open_tepes_9n = os.path.join(real_data_path, 'openTEPES_EAPP_2030_sc01_st1.mps')
gams_data_path = os.path.join(parent_path, 'data/GAMS_library_modified')
gams_model = os.path.join(gams_data_path, 'DINAM.mps')

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

def filter_variables(model):
    """
    Filter the variables and separate into basis or non basis variables
    """
    model.setParam('OutputFlag', 0)  
    model.optimize()

    # Get basis status for variables
    basis_gurobi = np.array(model.getAttr('vBasis'))  

    # Extract variable names
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)

    # Convert to NumPy array for efficient indexing
    variable_names = np.array(variable_names, dtype=object)  

    # Identify indices
    pos_basis = np.where(basis_gurobi == 0)[0]
    pos_non_basis = np.where(basis_gurobi != 0)[0]

    # Assign variables
    basis_variables = variable_names[pos_basis].tolist()
    non_basis_variables = variable_names[pos_non_basis].tolist()

    # Grouping variables
    names_var, asos_elements, group_var, invert_var = groups_by_variables(model)

    # Initialize group dictionaries
    basis_group = dict.fromkeys(group_var.keys(), 0)
    non_basis_group = dict.fromkeys(group_var.keys(), 0)

    # Count occurrences in basis groups
    for var in basis_variables:
        group_name = invert_var.get(var, None)  # Use .get() to avoid KeyError
        if group_name is not None:
            basis_group[group_name] += 1

    # Count occurrences in non-basis groups
    for bar in non_basis_variables:
        group_name = invert_var.get(bar, None)
        if group_name is not None:
            non_basis_group[group_name] += 1

    # Print summary
    print(f"Matrix shape: {A.shape}")
    print(f"Total basis variables: {len(basis_variables)}")
    print("Basis group counts:", basis_group)
    print("Non-basis group counts:", non_basis_group)
    return basis_variables, non_basis_variables 

def filter_constraints(model):
    model.setParam('OutputFlag', 0) 
    model.optimize()
    basis_constrs = model.getAttr('cBasis')
    basis_constrs =np.array(basis_constrs)
    A,b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    # print the number of basis constr = -1
    print(np.sum(basis_constrs == -1))
    # Get the names of the constraints that are in the basis
    basis_constrs = np.where(basis_constrs == 0)[0]
    constr_basis_names = [constraint_names[i] for i in basis_constrs]

def importance_variables_post(model):   
    basis_var, non_basis_var = filter_variables(model)  
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    # The metric for the basis var is the coefficient of the variable in the objective function
    c_array = np.array(c)
    # In variable name get the index of the variable that is in the basis
    variable_index_map = {var: i for i, var in enumerate(variable_names)}
    basis_var_index = [variable_index_map[var] for var in basis_var]
    non_basis_var_index = [variable_index_map[var] for var in non_basis_var]
    basis_var_metric = c_array[basis_var_index]
    print('Basis var metric: ',basis_var_metric[0:10])
    # Metric for the non basis is the reduced cost
    model.setParam('OutputFlag', 0) 
    model.optimize()
    reduced_cost =  list(model.getAttr('RC', model.getVars()))
    
    print('Reduced cost: ',reduced_cost[0:10])  
    reduced_cost = np.array(reduced_cost)

    non_basis_var_metric = reduced_cost[non_basis_var_index]
    # sort the non basis var metric
    non_basis_var_metric = np.abs(non_basis_var_metric)
    non_basis_var = non_basis_var.sort()    
    print('Non basis var metric: ',non_basis_var_metric[0:10])  
    return basis_var_index
    
def filter_constraints_standar(model):
    """
    Filter the constraints from an standard model
    """
    model.setParam('OutputFlag', 0)
    model.optimize()
    basis_constrs = model.getAttr('cBasis')
    basis_constrs = np.array(basis_constrs)
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    basis_var, non_basis_var = filter_variables(model)  
    slack_var = [var for var in basis_var if var.startswith('slack')]
    non_binding_constr = [var.split('_', 1)[1] for var in slack_var]
    non_binding_set = set(non_binding_constr)  # Convert to set for fast lookups
    binding_constr = [constr for constr in constraint_names if constr not in non_binding_set]
    non_binding_constr = list(non_binding_set)
    names_constr, asos_elements, group_constr, invert_constr = groups_by_constraints(model)
    binding_group = dict.fromkeys(group_constr.keys(), 0)
    non_binding_group = dict.fromkeys(group_constr.keys(), 0)

    for constr in binding_constr:
        group_name = invert_constr.get(constr, None)
        if group_name is not None:
            binding_group[group_name] += 1
    
    for constr in non_binding_constr:
        group_name = invert_constr.get(constr, None)
        if group_name is not None:
            non_binding_group[group_name] += 1
    print('Binding group counts: ',binding_group)
    print('Non binding group counts ',non_binding_group)
    return binding_constr, non_binding_constr


if __name__ == '__main__':
    model = gp.read(gams_model)
    #model = set_real_objective(model)
    model_standard = standard_form_e2(model)
    model_canonical = canonical_model(model)   
    #filter_variables(model_standard)
    #filter_constraints_standar(model_standard)
    importance_variables_post(model_standard)