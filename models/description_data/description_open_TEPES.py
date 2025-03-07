import os 
import gurobipy as gp
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

"""
IMPORTANT:
Thiis code is thinking to be used in the description of the data in the openTEPES model.
"""

# PATH TO THE DATA
parent_path = os.path.dirname(os.getcwd())
root_interpretable_optimization = os.path.dirname(parent_path)


# IMPORT FILES THAT ARE IN THE ROOT DIRECTORY
sys.path.append(parent_path)
from  models.utils_models.utils_functions import *
from models.utils_models.standard_model import *
from .plot_matrix import *

# PATH TO THE DATA
real_data_path = os.path.join(parent_path, 'data/real_data')
open_tepes_9n = os.path.join(real_data_path, 'openTEPES_EAPP_2030_sc01_st1.mps')

# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/global/sparse/real_problems') 
results_sparsification_open_tepes_9n = os.path.join(results_folder, 'sparsification_improve/epsilon_sparsification_openTEPES_EAPP_2030_sc01_st1_flexibility1.json')
results_cols_open_tepes_9n = os.path.join(results_folder, 'epsilon_cols_abs/epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')
results_rows_open_tepes_9n = os.path.join(results_folder, 'epsilon_rows_abs/epsilon_rows_openTEPES_EAPP_2030_sc01_st1.json')

figures_folder = os.path.join(parent_path, 'figures_new/description/openTEPES_models') 
figures_open_tepes_9n = os.path.join(figures_folder, 'openTEPES_EAPP_2030_sc01_st1')



def groups_by_variables(model):
    """
    Process and openTEPES model to get the groups of variables
    Parameters:
    - model: The optimization model to analyze.
    Returns:
    - names: The names of the variables in the model.
    - associated_elements: The number of associated elements for each group.
    - group_dict: A dictionary mapping group names to variable names.
    - inverted_group_dict: A dictionary mapping variable names to group names.
    """
    # Retrieve variables and their names
    variables = model.getVars()
    names = [var.VarName for var in variables]
    group_vars = [name.split('(')[0] for name in names]
    unique_groups = set(group_vars)

    # Create a dictionary to store variables by group
    group_dict = {group: [] for group in unique_groups}
    for name in names:
        group = name.split('(')[0]
        group_dict[group].append(name)

    # Retrieve the constraint matrix A as CSR for efficient row and column operations
    A = model.getA().tocsc()

    name_to_index = {name: idx for idx, name in enumerate(names)}

    # Calculate associated elements efficiently
    associated_elements = {group: 0 for group in unique_groups}
    for group, var_names in group_dict.items():
        group_indices = [name_to_index[var] for var in var_names]
        non_zero_counts = (A[:, group_indices] != 0).sum(axis=0).A1
        associated_elements[group] += non_zero_counts.sum()

    # Combine all groups starting with 'slack' into a single key
    slack_combined = []
    slack_count = 0
    for group in list(group_dict.keys()):
        if group.startswith('slack'):
            slack_combined.extend(group_dict.pop(group))
            slack_count += associated_elements.pop(group, 0)
    if slack_combined:
        group_dict['slack'] = slack_combined
        associated_elements['slack'] = slack_count

    groups_to_rename = [group for group in group_dict if group.startswith('slack')]
    
    # Create an inverted group dictionary
    inverted_group_dict = {name: name.split('(')[0] for name in names}
    for group in groups_to_rename:
        inverted_group_dict.update({name: 'slack' for name in group_dict[group]})

    return names, associated_elements, group_dict, inverted_group_dict
    
def groups_by_constraints(model):
    """
    Get the names of the constraints in the model.
    Parameters:
    - model: The optimization model to analyze.
    Returns:
    - names: The names of the constraints in the model.
    - associated_elements: The number of associated elements for each group. 
    - group_dict: A dictionary mapping group names to constraint names. 
    - inverted_group_dict: A dictionary mapping constraint names to group names. 
    """
    constraints = model.getConstrs()
    names = [constr.ConstrName for constr in constraints]
    groups_names = [name.split('(')[0] for name in names]
    unique_groups = set(groups_names)
  
    group_dict = {group: [] for group in unique_groups}
    name_to_index = {name: idx for idx, name in enumerate(names)}

    for name in names:
        group = name.split('(')[0]
        group_dict[group].append(name)

    groups_to_rename = [group for group in group_dict if group.startswith('v')]
        
    for group in groups_to_rename:
            group_dict[f'{group}_bound'] = group_dict.pop(group)
    associated_elements = {group: 0 for group in group_dict.keys()}

    A = model.getA().tocsc()
    for group, const_names in group_dict.items(): 
        group_indices = [name_to_index[constr] for constr in const_names]
        non_zero_counts = (A[group_indices, :] != 0).sum(axis=1).A1
        associated_elements[group] = non_zero_counts.sum()

    inverted_group_dict = {name: name.split('(')[0] for name in names}
    for group in groups_to_rename:
        inverted_group_dict.update({name: f'{group}_bound' for name in group_dict[f'{group}_bound']})
        
    return names, associated_elements, group_dict, inverted_group_dict

def get_name_variable_constraint(A, variable_names, constraint_names, index_row, index_col):
    """
    Get the name of the variable and the constraint of a matrix
    """
    # Directly access the names using the indices
    name_variable = variable_names[index_col]
    name_constraint = constraint_names[index_row]
    return name_variable, name_constraint

def map_position_to_names(model):
    """
    Map positions in model.getA() to variable and constraint names.
    
    Parameters:
    - model: The optimization model with getA() support.
    
    Returns:
    - position_to_names: A dictionary mapping (i, j) positions in A to (constraint_name, variable_name).
    """
    # Get the matrix A, constraints, and variables
    A = model.getA()  
    constraints = model.getConstrs() 
    variables = model.getVars() 

    # Get names of constraints and variables
    constraint_names = [constr.ConstrName for constr in constraints]
    variable_names = [var.VarName for var in variables]

    # Map positions in A to names
    position_to_names = {}
    A_coo = A.tocoo()  

    for i, j, value in zip(A_coo.row, A_coo.col, A_coo.data):  
        position_to_names[(i, j)] = (constraint_names[i], variable_names[j])
    return position_to_names


def get_total_by_group(dict_to_analyze): 
    """
    Get the total of elements in the group.
    """
    count_by_group = {}
    for key, value in dict_to_analyze.items():
        count_by_group[key] = value
    return count_by_group

def relation_equations(model, metric = 'sum'):
    """
    Get the relation between the equations and the variables
    """
    names_var, associated_variables, group_variables, inverted_group_v = groups_by_variables(model)
    names_constr, associated_constraints, group_constraints, inverted_group_c = groups_by_constraints(model)
    A = model.getA()  # Sparse matrix representation of the constraints
    constraints = model.getConstrs()  # List of constraints
    variables = model.getVars()  # List of variables

    # Get names of constraints and variables
    constraint_names = [constr.ConstrName for constr in constraints]
    variable_names = [var.VarName for var in variables]

    # Map positions in A to names
    position_to_names = {}
    postion_to_value = {}   
    A_coo = A.tocoo()  # Convert sparse matrix A to COOrdinate format for iteration

    for i, j, value in zip(A_coo.row, A_coo.col, A_coo.data):  # Iterate over non-zero entries
        #if value != 0:  # Only map non-zero entries
        position_to_names[(i, j)] = (constraint_names[i], variable_names[j])
        postion_to_value[(i, j)] = A[i, j]
    # For each group sum the values related to the group of variables and constraints
    group_variables = list(group_variables.keys())
    group_constraints = list(group_constraints.keys())
    asos_coefs =  {(group_constr, group_var): 0 for group_constr in group_constraints for group_var in group_variables}
    for position, value in postion_to_value.items():
        name_var = position_to_names[position][1]
        name_constr = position_to_names[position][0]
        group_var = inverted_group_v.get(name_var)
        group_constr = inverted_group_c.get(name_constr)
        if group_var and group_constr:
            asos_coefs[(group_constr, group_var)] += value
    elements_groups = create_association_dict(group_variables, group_constraints, inverted_group_v, inverted_group_c, position_to_names)
    if metric == 'mean':
        asos_coefs = {key: value / elements_groups[key] for key, value in asos_coefs.items() if elements_groups[key] != 0}
    plot_group_matrix(group_variables, group_constraints, asos_coefs, f'Relation of the equation per groups ({metric})', 2, None)

def create_association_dict(group_variables, group_constraints, inverted_variable_groups, inverted_constraint_groups, postoname):
    """
    Create a dictionary with the association of variables and constraints
    Parameters:
    - group_variables: The groups of variables.
    - group_constraints: The groups of constraints.
    - inverted_variable_groups: The inverted dictionary of variables.
    - inverted_constraint_groups: The inverted dictionary of constraints.
    - postoname: The position to name dictionary.
    Returns:
    - association_dict: A dictionary with the association of variables and constraints. The elements of the (group_constraint, group_variable) are the number of elements associated.
    """
    association_dict = {(group_constr, group_var): 0 for group_constr in group_constraints for group_var in group_variables}
    for posiition, name in postoname.items():
        name_var = name[1]
        name_constr = name[0]
        group_var = inverted_variable_groups.get(name_var)
        group_constr = inverted_constraint_groups.get(name_constr)
        if group_var.startswith('slack'):
            group_var = 'slack'
        if group_var and group_constr:
            association_dict[(group_constr, group_var)] += 1
        else:
            print('No group found for:', name_var, name_constr)
    return association_dict

def number_elements_groups(model):
    """
    Get the number of elements in the groups of variables and constraints.
    """
    names_var, associated_variables, group_variables, inverted_group_v = groups_by_variables(model)
    names_constr, associated_constraints, group_constraints, inverted_group_c = groups_by_constraints(model)
    pos_to_name = map_position_to_names(model)
    elements_group_constrs_and_vars = create_association_dict(group_variables, group_constraints, inverted_group_v, inverted_group_c, pos_to_name)
    plot_group_matrix(group_variables, group_constraints, elements_group_constrs_and_vars, 'Number of elements per group of constraints and variables', 0, figures_open_tepes_9n)
    return elements_group_constrs_and_vars


if __name__ == '__main__':

    model_open_tepes = gp.read(open_tepes_9n)
    model_standar_open_tepes = standard_form_e2(model_open_tepes)
    relation_equations(model_standar_open_tepes, 'mean')
    # names_var, associated_variables, group_variables, inverted_group_v = groups_by_variables(model_standar_open_tepes)
    # names_constr, associated_constraints, group_constraints, inverted_group_c = groups_by_constraints(model_standar_open_tepes) 
    # plot_histogram(group_variables, 'variables', 'Number of associated elements per group of variables', figures_open_tepes_9n)
    # plot_histogram(group_constraints, 'constraints', 'Number of associated elements per group of constraints', figures_open_tepes_9n)