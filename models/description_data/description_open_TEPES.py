import os 
import gurobipy as gp
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/global/sparse/real_problems') 
results_sparsification_open_tepes_9n = os.path.join(results_folder, 'sparsification/epsilon_sparsification_openTEPES_EAPP_2030_sc01_st1.json')
results_cols_open_tepes_9n = os.path.join(results_folder, 'epsilon_cols_abs/epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')

def read_json(file):
    """
    Read the json file
    """
    with open(file) as f:
        data = json.load(f)

    primal_model = data['openTEPES_EAPP_2030_sc01_st1']['primal']
    rows_changed = primal_model['rows_changed']
    columns_changed = primal_model['columns_changed']
    indices_changed = primal_model['changed_indices']   
    epsilons = primal_model['epsilon']
    obj_value = primal_model['objective_function']
    return rows_changed, columns_changed, indices_changed, epsilons, obj_value

def types_variables(model):
    """
    Get the types of the variables in the model
    """
    variables = model.getVars()
    types = [var.VType for var in variables]
    return types

def variable_names(model): 
    """
    Get the names of the variables in the model
    """
    variables = model.getVars()
    names_vars = [var.VarName for var in variables]
    return names_vars

def constraints_names(model): 
    """
    Get the names of the constraints in the model
    """
    constraints = model.getConstrs()
    names_constr = [constr.ConstrName for constr in constraints]
    return names_constr

def groups_by_variables(model):
    """
    Efficiently process variables, group them, and calculate associated elements.
    """
    # Retrieve variables and their names
    variables = model.getVars()
    names = [var.VarName for var in variables]

    # Group variable names by their prefix (text before '(')
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
        # Get indices of variables in this group
        group_indices = [name_to_index[var] for var in var_names]

        # Sum non-zero elements for all variables in the group
        non_zero_counts = (A[:, group_indices] != 0).sum(axis=0).A1

        # Update associated elements in a single step
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

    # Create an inverted group dictionary
    inverted_group_dict = {name: name.split('(')[0] for name in names}

    # Return processed data
    return names, associated_elements, group_dict, inverted_group_dict



def groups_by_constraints(model):
    """
    Get the names of the constraints in the model.
    """
    constraints = model.getConstrs()
    names = [constr.ConstrName for constr in constraints]
    groups_names = [name.split('(')[0] for name in names]
    unique_groups = set(groups_names)
    
    # Create a dictionary to store the constraints by group
    group_dict = {group: [] for group in unique_groups}
    name_to_index = {name: idx for idx, name in enumerate(names)}
    # Populate the dictionary with constraints grouped by their names
    for name in names:
        group = name.split('(')[0]
        group_dict[group].append(name)

    associated_elements = {group: 0 for group in unique_groups}
    A = model.getA().tocsc()
    for group, const_names in group_dict.items(): 
        group_indices = [name_to_index[constr] for constr in const_names]
        non_zero_counts = (A[group_indices, :] != 0).sum(axis=1).A1
        associated_elements[group] = non_zero_counts.sum()
    # Group the groups that stat with slack in the dict
    inverted_group_dict = {name: name.split('(')[0] for name in names}
        
    return names, associated_elements, group_dict, inverted_group_dict

# Recuperar el nombre de la variable y la restricción de una matriz
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
    A = model.getA()  # Sparse matrix representation of the constraints
    constraints = model.getConstrs()  # List of constraints
    variables = model.getVars()  # List of variables

    # Get names of constraints and variables
    constraint_names = [constr.ConstrName for constr in constraints]
    variable_names = [var.VarName for var in variables]

    # Map positions in A to names
    position_to_names = {}
    A_coo = A.tocoo()  # Convert sparse matrix A to COOrdinate format for iteration

    for i, j, value in zip(A_coo.row, A_coo.col, A_coo.data):  # Iterate over non-zero entries
        #if value != 0:  # Only map non-zero entries
        position_to_names[(i, j)] = (constraint_names[i], variable_names[j])
    return position_to_names

def indices_changes_analysis(model, rows_changed, epsilon_number): 
    """
    Get the indices of the changes in the rows
    """
    # Get the indices of the changes in the rows
    indices_changes = [(i, j) for i, j in enumerate(rows_changed)]
    names_var, associated_variables, groups_var, inverted_group_var = groups_by_variables(model)
    names_constrs, associated_constraints, groups_constraints, inverted_group_constr = groups_by_constraints(model)
    map_position = map_position_to_names(model)
    first_index = indices_changes[epsilon_number][1]
    # Initialize dictionaries with zero values
    changed_groups_var = dict.fromkeys(groups_var, 0)
    changed_groups_constraints = dict.fromkeys(groups_constraints, 0)
    # Process elements in the first index
    for element in first_index:
        name_constraint, name_variable = map_position[tuple(element)]

        associated_group_var = inverted_group_var.get(name_variable)
        associated_group_constr = inverted_group_constr.get(name_constraint)
        # print(associated_group_var, associated_group_constr)
        if associated_group_var:
            changed_groups_var[associated_group_var] += 1
        if associated_group_constr:
            changed_groups_constraints[associated_group_constr] += 1
        if 'slack' in associated_group_var:
            changed_groups_var['slack'] += 1
    # Put a warning if the changed group supperates the total of the group
    return changed_groups_var, changed_groups_constraints, associated_variables, associated_constraints

def main(model, epsilon_number, operation): 
    """
    Get the main variables that are eliminated by the presolve method
    """
    if operation == 'Sparsification': 
        _, _, indices_changed, epsilons, obj_value = read_json(results_sparsification_open_tepes_9n)
    elif operation == 'Columns': 
        _, columns_changed, _, epsilons, _ = read_json(results_cols_open_tepes_9n)
    changed_groups_var, changed_group_constraints, associated_variables, associated_constraints = indices_changes_analysis(model, indices_changed, epsilon_number)
    return changed_groups_var, changed_group_constraints, associated_variables, associated_constraints, epsilons[epsilon_number], obj_value[epsilon_number]

def analyzes_groups(dict_to_analyze): 
    """
    Analyze groups by counting the elements in each group.
    
    Parameters:
    - group: The group to analyze.
    - type_group: The type of group (e.g., 'Variables' or 'Constraints').
    - dict_to_analyze: The dictionary containing the groups and their associated elements.
    
    Returns:
    - A dictionary with the count of elements for each group.
    """
    count_by_group = {}
    for key, value in dict_to_analyze.items():
        count_by_group[key] = value
    return count_by_group

def get_total_by_group(dict_to_analyze): 
    """
    Get the total of elements in the group
    """
    count_by_group = {}
    for key, value in dict_to_analyze.items():
        count_by_group[key] = value
    return count_by_group

def plot_changes_histogram_with_slider(model): 
    """
    Interactive plot with a slider to dynamically update plots based on EPSILON_NUMBER.

    Parameters:
    - model: The model to analyze.
    - group_variables: The variable groups in the model.
    - group_constraints_original: The constraint groups in the model.
    """
    # Initial EPSILON_NUMBER
    epsilon_number = 1

    # Initial data computation
    changed_group_var, changed_group_constraints, associated_variables, associated_constraints, epsilon, obj_value = main(model, epsilon_number, 'Sparsification')
    count_by_group_var = analyzes_groups(changed_group_var)
    count_by_group_constraints = analyzes_groups(changed_group_constraints)
    sum_changes = sum(group for group in changed_group_var.values())
    total_variables = get_total_by_group(associated_variables)
    total_constraints = get_total_by_group(associated_constraints)
    # Initial proportions
    groups_var = list(total_variables.keys())
    groups_var.remove('slack')
    groups_constraints = list(total_constraints.keys())
    # Sort the groups alphabetically
    groups_var.sort()
    groups_constraints.sort()
    # Create the figure for variables and constraints
    fig, (ax_var, ax_constraints) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.4)  # Increase space between subplots

    x_var = np.arange(len(groups_var))
    total_heights_var = [total_variables[group] for group in groups_var]
    change_heights_var = [count_by_group_var[group] for group in groups_var]
    bar_total_var = ax_var.bar(x_var, total_heights_var, color="blue", alpha=0.5, label="Total Variables")
    bar_changed_var = ax_var.bar(x_var, change_heights_var, color="red", alpha=0.7, label="Changed Variables")
    ax_var.set_title("Changes in Variables by Group")
    ax_var.set_xlabel("Groups")
    ax_var.set_ylabel("Proportion of Changes")
    ax_var.set_xticks(x_var)
    ax_var.set_xticklabels(groups_var, rotation=45, ha="right")

    # Plot constraints
    x_constraints = np.arange(len(groups_constraints))
    total_heights_constraints = [
        total_constraints.get(group, 0) for group in groups_constraints
    ]
    change_heights_constraints = [
        count_by_group_constraints.get(group, 0)
        for group in groups_constraints
    ]

    bar_total_constraints = ax_constraints.bar(x_constraints, total_heights_constraints, color="blue", alpha=0.5, label="Total Constraints")
    bar_changed_constraints = ax_constraints.bar(x_constraints, change_heights_constraints, color="red", alpha=0.7, label="Changed Constraints")

    ax_constraints.set_title("Changes in Constraints by Group")
    ax_constraints.set_xlabel("Groups")
    ax_constraints.set_ylabel("Proportion of Changes")
    ax_constraints.set_xticks(x_constraints)
    ax_constraints.set_xticklabels(groups_constraints, rotation=45, ha="right")

    # Display objective value and epsilon in the top-right corner
    fig.text(0.95, 0.95, f"Epsilon: {epsilon:.4f}\nObjective: {obj_value:.4f}",
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    # Create a slider
    ax_slider = plt.axes([0.2, 0.92, 0.5, 0.03], facecolor='lightgoldenrodyellow')  # Slider at the top
    epsilon_slider = Slider(ax_slider, "EPSILON_NUMBER", 1, 17, valinit=epsilon_number, valstep=1)

    # Update function
    def update(val):
        epsilon_number = int(epsilon_slider.val)
        changed_group_var, changed_group_constraints,associated_variables, associated_constraints,  epsilon, obj_value = main(model, epsilon_number, 'Sparsification')
        count_by_group_var = analyzes_groups(changed_group_var)
        count_by_group_constraints = analyzes_groups(changed_group_constraints)

        # Update total and changed heights for variables and constraints
        total_heights_var = [total_variables[group] for group in groups_var]
        change_heights_var = [
            count_by_group_var.get(group, 0)
            for group in groups_var
        ]

        total_heights_constraints = [total_constraints[group] for group in groups_constraints]
        change_heights_constraints = [
            count_by_group_constraints.get(group, 0)
            for group in groups_constraints
        ]

        # Update bars for variables
        for rect, new_total_height in zip(bar_total_var, total_heights_var):
            rect.set_height(new_total_height)
        for rect, new_change_height in zip(bar_changed_var, change_heights_var):
            rect.set_height(new_change_height)

        # Update bars for constraints
        for rect, new_total_height in zip(bar_total_constraints, total_heights_constraints):
            rect.set_height(new_total_height)
        for rect, new_change_height in zip(bar_changed_constraints, change_heights_constraints):
            rect.set_height(new_change_height)

        # Update text box
        fig.texts[-1].set_text(f"Epsilon: {epsilon:.4f}\nObjective: {obj_value:.4f}")

        # Redraw the figure
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    epsilon_slider.on_changed(update)

    plt.show()
if __name__ == '__main__':
    model_open_tepes = gp.read(open_tepes_9n)
    model_standar_open_tepes = standard_form_e1(model_open_tepes)
    # rows_changed, columns_changed, indices_changed, epsilon = read_json(results_sparsification_open_tepes_9n)
    # # Get the different types of variables in the model
    # types = types_variables(model_standar_open_tepes)
    # names_constraints_original, associated_constraints, group_constraints_original, inverted_group_c = groups_by_constraints(model_standar_open_tepes)
    # # print('hecho')
    # names_variables, associated_variables, group_variables, inverted_group_v = groups_by_variables(model_standar_open_tepes)
    # print('hecho')
    # model = model_standar_open_tepes
    # group_variables = group_variables
    # #group_constraints = group_constraints_original
    # A, _, _, _, _, _, _, _, _ = get_model_matrices(model)
    # print('Model matrices shape:', A.shape) 
    # # Total variables sum the total of variables in each group
    # total_variables = get_total_by_group(associated_variables)
    # #total_constraints = get_total_by_group(group_constraints)
    # total_v = sum(total_variables.values())
    # #total_c = sum(total_constraints.values())
    # print('Total variables:', total_v)
    #print('Total constraints:', total_c)
    plot_changes_histogram_with_slider(model_standar_open_tepes) # , associated_variables, associated_constraints)