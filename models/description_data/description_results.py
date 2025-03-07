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
from plot_matrix import *
from description_open_TEPES import *

# PATH TO THE DATA
real_data_path = os.path.join(parent_path, 'data/real_data')
open_tepes_9n = os.path.join(real_data_path, 'openTEPES_EAPP_2030_sc01_st1.mps')

# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/global/sparse/real_problems') 
results_sparsification_open_tepes_9n = os.path.join(results_folder, 'sparsification_improve/epsilon_sparsification_openTEPES_EAPP_2030_sc01_st1_flexibility1.json')
results_cols_open_tepes_9n = os.path.join(results_folder, 'epsilon_cols_abs/epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')
results_rows_open_tepes_9n = os.path.join(results_folder, 'epsilon_rows_abs/epsilon_rows_openTEPES_EAPP_2030_sc01_st1.json')

figures_folder = os.path.join(parent_path, 'figures_new/global/sparse/real_problems') 
figures_sparsifaction_open_tepes_9n = os.path.join(figures_folder, 'sparsification_improve')

def read_json_results(file):
    """
    Read the json file of the results for the openTEPES model
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

def indices_changes_analysis_v_c(model, indices_changed, epsilon_number):
    """
    Get the indices of the changes in the variable and the constraints groups_by_constraints
    """
    indices_changes = [(i, j) for i, j in enumerate(indices_changed)]
    indices_changes = indices_changes[epsilon_number][1]
    names_var, associated_variables, group_variables, inverted_group_v = groups_by_variables(model)
    names_constr, associated_constraints, group_constraints, inverted_group_c = groups_by_constraints(model)
    changes_elements = {}
    postoname = map_position_to_names(model)
    for element in indices_changes:
        # Create tuple name as key and value the name of constraint and variable
        name = tuple(element)
        name_var = postoname[name][1]
        name_constr = postoname[name][0]
        changes_elements[name] = (name_constr, name_var)
    return changes_elements

def main(model, epsilon_number, operation, plot='Histogram'): 
    """
    Get the main variables that are eliminated by the presolve method
    """
    if operation == 'Sparsification': 
        _, _, indices_changed, epsilons, obj_value = read_json_results(results_sparsification_open_tepes_9n)
    elif operation == 'Columns': 
        _, columns_changed, _, epsilons, _ = read_json_results(results_cols_open_tepes_9n)
    
    if plot == 'Histogram':
        changed_groups_var, changed_group_constraints, associated_variables, associated_constraints = indices_changes_analysis(model, indices_changed, epsilon_number)
        return changed_groups_var, changed_group_constraints, associated_variables, associated_constraints, epsilons[epsilon_number], obj_value[epsilon_number]
    elif plot == 'Matrix':
        changes_elements = indices_changes_analysis_v_c(model, indices_changed, epsilon_number)
        return changes_elements, epsilons, obj_value


def plot_changes_histogram_with_slider(model): 
    """
    Interactive plot with a slider to dynamically update plots based on EPSILON_NUMBER.

    Parameters:
    - model: The model to analyze.
    """
    # Initial EPSILON_NUMBER
    epsilon_number = 1

    # Initial data computation
    changed_group_var, changed_group_constraints, associated_variables, associated_constraints, epsilon, obj_value = main(model, epsilon_number, 'Sparsification')
    count_by_group_var = analyzes_groups(changed_group_var)
    count_by_group_constraints = analyzes_groups(changed_group_constraints)
    total_variables = get_total_by_group(associated_variables)
    total_constraints = get_total_by_group(associated_constraints)

    # Remove slack variables group if present
    groups_var = list(total_variables.keys())
    groups_var.remove('slack')
    groups_constraints = list(total_constraints.keys())
    groups_var.sort()
    groups_constraints.sort()

    # Create the figure for variables and constraints
    fig, (ax_var, ax_constraints) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.4)

    # Variables data
    x_var = np.arange(len(groups_var))
    total_heights_var = [total_variables[group] for group in groups_var]
    change_heights_var = [count_by_group_var.get(group, 0) for group in groups_var]
    percentages_var = [
        (change / total * 100 if total > 0 else 0)
        for change, total in zip(change_heights_var, total_heights_var)
    ]

    bar_total_var = ax_var.bar(x_var, total_heights_var, color="blue", alpha=0.5, label="Total Variables")
    bar_changed_var = ax_var.bar(x_var, change_heights_var, color="red", alpha=0.7, label="Changed Variables")
    ax_var.set_title("Changes in Variables by Group")
    ax_var.set_xlabel("Groups")
    ax_var.set_ylabel("Count")
    ax_var.set_xticks(x_var)
    ax_var.set_xticklabels(groups_var, rotation=45, ha="right")

    # Add percentage labels for variables
    percentage_texts_var = []
    for i, percentage in enumerate(percentages_var):
        text = ax_var.text(
            i, change_heights_var[i] + 0.5, f"{percentage:.1f}%",
            ha='center', va='bottom'
        )
        percentage_texts_var.append(text)

    # Constraints data
    x_constraints = np.arange(len(groups_constraints))
    total_heights_constraints = [total_constraints[group] for group in groups_constraints]
    change_heights_constraints = [count_by_group_constraints.get(group, 0) for group in groups_constraints]
    percentages_constraints = [
        (change / total * 100 if total > 0 else 0)
        for change, total in zip(change_heights_constraints, total_heights_constraints)
    ]

    bar_total_constraints = ax_constraints.bar(x_constraints, total_heights_constraints, color="blue", alpha=0.5, label="Total Constraints")
    bar_changed_constraints = ax_constraints.bar(x_constraints, change_heights_constraints, color="red", alpha=0.7, label="Changed Constraints")
    ax_constraints.set_title("Changes in Constraints by Group")
    ax_constraints.set_xlabel("Groups")
    ax_constraints.set_ylabel("Count")
    ax_constraints.set_xticks(x_constraints)
    ax_constraints.set_xticklabels(groups_constraints, rotation=45, ha="right")

        # Display objective value and epsilon in the top-right corner
    fig.text(0.95, 0.95, f"Epsilon: {epsilon:.4f}\nObjective: {obj_value:.4f}",
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    # Add percentage labels for constraints
    percentage_texts_constraints = []
    for i, percentage in enumerate(percentages_constraints):
        text = ax_constraints.text(
            i, change_heights_constraints[i] + 0.5, f"{percentage:.1f}%",
            ha='center', va='bottom'
        )
        percentage_texts_constraints.append(text)

    # Create a slider
    ax_slider = plt.axes([0.2, 0.92, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    epsilon_slider = Slider(ax_slider, "EPSILON_NUMBER", 1, 17, valinit=epsilon_number, valstep=1)

    # Update function
    def update(val):
        nonlocal percentage_texts_var, percentage_texts_constraints
        epsilon_number = int(epsilon_slider.val)
        changed_group_var, changed_group_constraints, associated_variables, associated_constraints, epsilon, obj_value = main(model, epsilon_number, 'Sparsification')
        count_by_group_var = analyzes_groups(changed_group_var)
        count_by_group_constraints = analyzes_groups(changed_group_constraints)

        # Update totals and changes
        total_heights_var = [total_variables[group] for group in groups_var]
        change_heights_var = [count_by_group_var.get(group, 0) for group in groups_var]
        percentages_var = [
            (change / total * 100 if total > 0 else 0)
            for change, total in zip(change_heights_var, total_heights_var)
        ]

        total_heights_constraints = [total_constraints[group] for group in groups_constraints]
        change_heights_constraints = [count_by_group_constraints.get(group, 0) for group in groups_constraints]
        percentages_constraints = [
            (change / total * 100 if total > 0 else 0)
            for change, total in zip(change_heights_constraints, total_heights_constraints)
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

        # Delete old percentage labels
        for text in percentage_texts_var:
            text.remove()
        percentage_texts_var.clear()

        for text in percentage_texts_constraints:
            text.remove()
        percentage_texts_constraints.clear()

        # Add new percentage labels for variables
        for i, percentage in enumerate(percentages_var):
            text = ax_var.text(
                i, change_heights_var[i] + 0.5, f"{percentage:.1f}%",
                ha='center', va='bottom'
            )
            percentage_texts_var.append(text)

        # Add new percentage labels for constraints
        for i, percentage in enumerate(percentages_constraints):
            text = ax_constraints.text(
                i, change_heights_constraints[i] + 0.5, f"{percentage:.1f}%",
                ha='center', va='bottom'
            )
            percentage_texts_constraints.append(text)
        
        fig.texts[-1].set_text(f"Epsilon: {epsilon:.4f}\nObjective: {obj_value:.4f}")
        # Redraw the figure
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    epsilon_slider.on_changed(update)

    plt.show()


def plot_slider_group_matrix(model):
    """
    Interactive heatmap plot with a slider to represent the percentage of changes
    in variable-constraint group associations based on epsilon.

    Parameters:
        model: The optimization model to analyze.

    Returns:
        None
    """
    # Initial epsilon number
    epsilon_number = 1

    # Get initial data
    names_var, associated_variables, group_variables, inverted_group_v = groups_by_variables(model)
    names_constr, associated_constraints, group_constraints, inverted_group_c = groups_by_constraints(model)
    map_position = map_position_to_names(model)

    total_variables = get_total_by_group(associated_variables)
    total_constraints = get_total_by_group(associated_constraints)

    groups_var = list(total_variables.keys())
    # if 'slack' in groups_var:
    #     groups_var.remove('slack')
    groups_constraints = list(total_constraints.keys())
    groups_var.sort()
    groups_constraints.sort()

    # Create association dictionary
    association_dict = create_association_dict(
        group_variables, group_constraints, inverted_group_v, inverted_group_c, map_position
    )

    # Calculate initial changes and percentages
    changes_pos_to_name, epsilons, obj_values = main(model, epsilon_number, 'Sparsification', 'Matrix')
    changes = create_association_dict(group_variables, groups_constraints, inverted_group_v, inverted_group_c, changes_pos_to_name)
    epsilon = epsilons[epsilon_number]
    obj_value = obj_values[epsilon_number]
    print(changes_pos_to_name == map_position)
    def calculate_percentages(total_dict, change_dict):
        percentages = {}
        for group_pair, total in total_dict.items():
            change = change_dict.get(group_pair, 0)
            percentages[group_pair] = (change / total * 100) if total > 0 else 0
        return percentages

    percentages = calculate_percentages(association_dict, changes)

    # Initialize the matrix for heatmap
    def build_matrix(groups_constraints, groups_var, percentages):
        matrix = np.zeros((len(groups_constraints), len(groups_var)))
        variable_index = {group: idx for idx, group in enumerate(groups_var)}
        constraint_index = {group: idx for idx, group in enumerate(groups_constraints)}

        for (constr_group, var_group), percent in percentages.items():
            if var_group in variable_index and constr_group in constraint_index:
                i = constraint_index[constr_group]
                j = variable_index[var_group]
                matrix[i, j] = percent

        return matrix

    matrix = build_matrix(groups_constraints, groups_var, percentages)

    # Create the figure and heatmap
    fig, ax = plt.subplots(figsize=(16, 12))  # Increased size
    cax = ax.matshow(matrix, cmap='viridis', vmin=0, vmax=100)
    #ax.set_aspect(0.5)
    # Add colorbar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(groups_var)))
    ax.set_yticks(np.arange(len(groups_constraints)))
    ax.set_xticklabels(groups_var, rotation=45, ha='left')
    ax.set_yticklabels(groups_constraints)

    # Label each cell
    percentage_texts = []
    for i in range(len(groups_constraints)):
        for j in range(len(groups_var)):
            text = ax.text(j, i, f"{matrix[i, j]:.1f}", va='center', ha='center', color='white')
            percentage_texts.append(text)

    fig.text(0.95, 0.95, f"Epsilon: {epsilon:.4f}\nObjective: {obj_value:.4f}",
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    plt.title('Percentage Changes in Constraint-Variable Group Associations')
    plt.xlabel('Variable Groups')
    plt.ylabel('Constraint Groups')
    plt.tight_layout()

    # Add slider
    ax_slider = plt.axes([0.2, 0.92, 0.3, 0.03], facecolor='lightgoldenrodyellow')
    epsilon_slider = Slider(ax_slider, "EPSILON_NUMBER", 1, len(epsilons), valinit=epsilon_number, valstep=1)

    # Update function for the slider
    def update(val):
        nonlocal percentage_texts
        epsilon_number = int(epsilon_slider.val)

        # Recalculate changes and percentages
        changes_pos_to_name, epsilons, obj_values = main(model, epsilon_number, 'Sparsification', 'Matrix')
        changes = create_association_dict(group_variables, groups_constraints, inverted_group_v, inverted_group_c, changes_pos_to_name)
        epsilon = epsilons[epsilon_number]
        obj_value = obj_values[epsilon_number]
        percentages = calculate_percentages(association_dict, changes)

        # Update the matrix
        matrix = build_matrix(groups_constraints, groups_var, percentages)

        # Update heatmap values
        cax.set_data(matrix)

        # Update percentage labels
        for text in percentage_texts:
            text.remove()
        percentage_texts.clear()

        for i in range(len(groups_constraints)):
            for j in range(len(groups_var)):
                text = ax.text(j, i, f"{matrix[i, j]:.1f}", va='center', ha='center', color='white')
                percentage_texts.append(text)

        # Update epsilon and objective value
        if fig.texts:
            fig.texts[-1].set_text(f"Epsilon: {epsilon:.4f}\nObjective: {obj_value:.4f}")
        # Redraw the figure
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    epsilon_slider.on_changed(update)

    plt.show()

def plot_changes_matrix(model): 
    """
    Plot the changes in the matrix
    """
    names_var, associated_variables, group_variables, inverted_group_v = groups_by_variables(model)
    names_constr, associated_constraints, group_constraints, inverted_group_c = groups_by_constraints(model)
    map_position = map_position_to_names(model)
    changed_group_var, changed_group_constraints, associated_variables, associated_constraints, epsilon, obj_value = main(model, 1, 'Sparsification')
    count_by_group_var = get_total_by_group(changed_group_var)
    count_by_group_constraints = get_total_by_group(changed_group_constraints)
    total_variables = get_total_by_group(associated_variables)
    total_constraints = get_total_by_group(associated_constraints)
    groups_var = list(total_variables.keys())
    groups_var.remove('slack')
    groups_constraints = list(total_constraints.keys())
    groups_var.sort()
    groups_constraints.sort()
    association_dict = create_association_dict(group_variables, group_constraints, inverted_group_v, inverted_group_c, map_position)
    changes_pos_to_name, epsilon, obj_value = main(model, 1, 'Sparsification', 'Matrix')
    changes = create_association_dict(group_variables, groups_constraints, inverted_group_v, inverted_group_c, changes_pos_to_name)
    plot_group_matrix(group_variables, group_constraints, changes, 'Changes in sparsification', 3, figures_sparsifaction_open_tepes_9n)


if __name__ == '__main__':
    model = gp.read(open_tepes_9n)
    plot_slider_group_matrix(model)