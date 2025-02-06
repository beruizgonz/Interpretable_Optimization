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

# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/marginal_values/real_problems')
results_simplified_constraints = os.path.join(results_folder, 'simplified_constraints_epsilon/importance_openTEPES_EAPP_2030_sc01_st1_constraints.json')
results_simplified_variables = os.path.join(results_folder, 'simplified_variables_percentage/openTEPES_EAPP_2030_sc01_st1_variables.json')

# Save paths 
interactive_figures = os.path.join(parent_path, 'figures_new/interactive_figures/marginal_value/simplified_constraints_percentage')
real_problems = os.path.join(interactive_figures, 'real_problems')


def read_json(file):
    """
    Read the json file
    """
    with open(file) as f:
        data = json.load(f)

    percentage = data['thresholds']
    objective_values = data['objective_values']
    names_remove = data['names_remove']
    return percentage, objective_values, names_remove

def remove_by_group(model, percentage, names_remove, entity_type = 'constraints'): 
    if entity_type == 'constraints':
        names, asosiations, groups, inverted_groups = groups_by_constraints(model)
    elif entity_type == 'variables':
        names, asosiations, groups, inverted_groups = groups_by_variables(model)
    # Get the group of the variables I have remove
    group_remove = dict.fromkeys(groups.keys(), 0)
    removes = names_remove[percentage]
    for name in removes:
        group = inverted_groups[name]
        group_remove[group] += 1
    return asosiations, group_remove 

def remove_group(model, group, entity_type):
    # Remove all the constraints or variables of the group
    if entity_type == 'constraints':
        names, asosiations, groups, inverted_groups = groups_by_constraints(model)
    elif entity_type == 'variables':
        names, asosiations, groups, inverted_groups = groups_by_variables(model)
    for name in groups[group]:
        model.remove(model.getConstrByName(name))
    model.update()
    return model

def remove_all_groups(model, min_threshold, max_threshold, step, entity_type = 'constraints',  save_path = None):
    model_imp = normalize_variables(model)
    if entity_type == 'constraints':
        importance_constrs, constrs_names = constraints_importance(model_imp)
        group_imp = importances_by_groups(importance_constrs, constrs_names, model_imp, 'constraints')
    elif entity_type == 'variables':
        importance_vars, vars_names = variables_importance(model_imp)
        group_imp = importances_by_groups(importance_vars, vars_names, model_imp, 'variables')
    thresholds = []
    print(sorted(group_imp.items(), key=lambda x: x[1]))
    # threshold = min_threshold
    # while threshold <= max_threshold:
    #     thresholds.append(threshold)
    #     threshold += step
    thresholds = np.linspace(min_threshold, max_threshold, 20)
    print(thresholds)
    groups_remove = []
    for threshold in thresholds:
        print(f'Threshold: {threshold}')
        simplified_model = model.copy()
        for group in group_imp.keys():
            if int(group_imp[group]) <= threshold:
                groups_remove.append(group)
                simplified_model = remove_group(simplified_model, group, entity_type)
        simplified_model.setParam('OutputFlag', 0)  
        simplified_model.optimize()
        # Print the groups that remain  
        groups = list(group_imp.keys())
        groups_remain = [group for group in groups if group not in groups_remove]
        print(f'Groups that remain: {groups_remain}')
        print(f'Objective value: {simplified_model.objVal}')
    return simplified_model

def main_remove_all_groups(model, min_threshold, max_threshold, step, entity_type = 'constraints', save_path = None):
    model = standard_form_e2(model)
    group_remove = remove_all_groups(model, min_threshold, max_threshold, step, entity_type, save_path)
    print(group_remove)

def plot_changes_histogram_with_slider(model):
    """
    Interactive plot with a slider to dynamically update plots based on EPSILON_NUMBER.

    Parameters:
    - model: The model to analyze.
    """
    # Initial EPSILON_NUMBER
    epsilon_number = 0
    percentage, objective_values, names_remove = read_json(results_simplified_constraints)

    # Initial data computation
    associated_constraints, group_remove = remove_by_group(model, epsilon_number, names_remove, entity_type='constraints')
    total_constraints = get_total_by_group(associated_constraints)

    groups_constraints = list(total_constraints.keys())
    groups_constraints.sort()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)  # Give space for the slider

    # Constraints data
    x_constraints = np.arange(len(groups_constraints))
    total_heights_constraints = [total_constraints[group] for group in groups_constraints]
    change_heights_constraints = [group_remove.get(group, 0) for group in groups_constraints]
    percentages_constraints = [
        (change / total * 100 if total > 0 else 0)
        for change, total in zip(change_heights_constraints, total_heights_constraints)
    ]
    
    # Plot bars
    bar_total_constraints = plt.bar(x_constraints, total_heights_constraints, color="blue", alpha=0.5, label="Total Constraints")
    bar_changed_constraints = plt.bar(x_constraints, change_heights_constraints, color="red", alpha=0.7, label="Changed Constraints")

    # Labels & title
    plt.title("Changes in Constraints by Group")
    plt.xlabel("Groups")
    plt.ylabel("Count")
    plt.xticks(x_constraints, groups_constraints, rotation=45, ha="right")  # Fix labels
    plt.legend(loc="upper right")  # Move legend to avoid overlapping

    # Add percentage labels inside bars
    percentage_texts_constraints = []
    for i, perc in enumerate(percentages_constraints):
        text = ax.text(
            i, change_heights_constraints[i] + 0.5, f"{perc:.1f}%",
            ha='center', va='bottom' if change_heights_constraints[i] > 0 else 'top',
            fontsize=9, color='black'
        )
        percentage_texts_constraints.append(text)

    # Create a slider
    ax_slider = plt.axes([0.2, 0.95, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    epsilon_slider = Slider(ax_slider, "Slider", 0, 10, valinit=epsilon_number * 5, valstep=1)

    # Display objective value & percentage next to slider
    text_info = fig.text(
        0.85, 0.99, 
        f"Percentage: {percentage[epsilon_number]}% \n"
        f"Objective: {objective_values[epsilon_number]:.4f} \n"
        f"Total Constraints: {sum(total_heights_constraints)} \n"
        f"Changed Constraints: {sum(change_heights_constraints)}",
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )

    # Update function
    def update(val):
        nonlocal percentage_texts_constraints
        epsilon_number = int(epsilon_slider.val)
        associated_constraints, group_remove = remove_by_group(model, epsilon_number, names_remove, entity_type='constraints')

        # Update totals and changes
        total_heights_constraints = [total_constraints[group] for group in groups_constraints]
        change_heights_constraints = [group_remove.get(group, 0) for group in groups_constraints]
        percentages_constraints = [
            (change / total * 100 if total > 0 else 0)
            for change, total in zip(change_heights_constraints, total_heights_constraints)
        ]

        # Update bars for constraints
        for rect, new_total_height in zip(bar_total_constraints, total_heights_constraints):
            rect.set_height(new_total_height)
        for rect, new_change_height in zip(bar_changed_constraints, change_heights_constraints):
            rect.set_height(new_change_height)

        # Remove old percentage labels
        for text in percentage_texts_constraints:
            text.remove()
        percentage_texts_constraints.clear()

        # Add new percentage labels
        for i, perc in enumerate(percentages_constraints):
            text = ax.text(
                i, change_heights_constraints[i] + 0.5, f"{perc:.1f}%",
                ha='center', va='bottom' if change_heights_constraints[i] > 0 else 'top',
                fontsize=9, color='black'
            )
            percentage_texts_constraints.append(text)

        # Update the text next to the slider
        text_info.set_text(f"Percentage: {percentage[epsilon_number]}%  \n "
                           f" Objective: {objective_values[epsilon_number]:.4f} \n"
                           f"Total Constraints: {sum(total_heights_constraints)} \n"
                           f"Changed Constraints: {sum(change_heights_constraints)}")

        # Redraw the figure
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    epsilon_slider.on_changed(update)
    # Show plot
    plt.show()


def plot_changes_histogram_html(model, json_file, e_type, importance_by_group, save_path  = None):
    """
    Creates an interactive bar chart with a slider using Plotly and saves it as an HTML file.

    Parameters:
    - model: The optimization model to analyze.
    - json_file: Path to the JSON file containing data.
    """
    # Load data from JSON file
    percentage, objective_values, names_remove = read_json(json_file)

    # Get constraints by group
    associated_constraints, group_remove = remove_by_group(model, 0, names_remove, entity_type = e_type)
    total_constraints = get_total_by_group(associated_constraints)

    groups_constraints = list(total_constraints.keys())
    groups_constraints.sort()
    # Order groups by importance of importance_by_group
    groups_constraints = sorted(groups_constraints, key=lambda g: importance_by_group.get(g, 0), reverse=True)
    # order by groups with the highest changes
    #groups_constraints = sorted(groups_constraints, key=lambda g: group_remove.get(g, 0), reverse=True)

    # Create lists to store total and changed constraints per percentage
    total_heights_constraints = []
    change_heights_constraints = []
    
    for i in range(len(percentage)):
        associated_constraints, group_remove = remove_by_group(model, i, names_remove, entity_type=e_type)
        total_constraints = get_total_by_group(associated_constraints)
        
        total_heights_constraints.append([total_constraints[group] for group in groups_constraints])
        change_heights_constraints.append([group_remove.get(group, 0) for group in groups_constraints])

    # Compute total changes per group across all percentage levels
    # Use only the first percentage (index 0) for sorting
#     first_percentage_changes = change_heights_constraints[0]  # Take first percentage changes

# # Sort groups based on the number of changes in the first percentage (descending)
#     groups_constraints = sorted(groups_constraints, key=lambda g: first_percentage_changes[groups_constraints.index(g)], reverse=True)

    # Initialize figure
    fig = go.Figure()

    # Add initial bar chart (first threshold level) - Stacked Bars
    fig.add_trace(go.Bar(
        x=groups_constraints,
        y=total_heights_constraints[0],
        name="Total Constraints",
        marker=dict(color="blue", opacity=0.5)
    ))

    fig.add_trace(go.Bar(
        x=groups_constraints,
        y=change_heights_constraints[0],
        name="Changed Constraints",
        marker=dict(color="red", opacity=0.7)
    ))

    # Add percentage labels inside the bars
    annotations = []
    for i, group in enumerate(groups_constraints):
        total_val = total_heights_constraints[0][i]
        change_val = change_heights_constraints[0][i]
        percentage_val = (change_val / total_val * 100) if total_val > 0 else 0
        
        annotations.append(dict(
            x=group, y=change_val + 2,  # Adjust for better visibility
            text=f"{percentage_val:.1f}%",
            showarrow=False, font=dict(size=12, color="black")
        ))

    # Information Box (Like Matplotlib)
    info_text = f"""
    <b>Percentage:</b> {percentage[0]}%<br>
    <b>Objective:</b> {objective_values[0]:.4f}<br>
    <b>Total Constraints:</b> {sum(total_heights_constraints[0])}<br>
    <b>Changed Constraints:</b> {sum(change_heights_constraints[0])}
    """

    # Add Text Box
    fig.add_annotation(
        x=0.8, y=1.15, xref="paper", yref="paper",
        text=info_text,
        showarrow=False,
        align="right",
        font=dict(size=12, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    steps = []
    for i in range(len(percentage)):
        updated_info_text = f"""
        <b>Percentage:</b> {percentage[i]}%<br>
        <b>Objective:</b> {objective_values[i]:.4f}<br>
        <b>Total Constraints:</b> {sum(total_heights_constraints[i])}<br>
        <b>Changed Constraints:</b> {sum(change_heights_constraints[i])}
        """
        annotations_step = []
        
        # Keep total constraints fixed (blue bars)
        total_fixed = total_heights_constraints[0]  # Blue bars remain the same
        change_dynamic = change_heights_constraints[i]  # Red bars update dynamically

        # Generate percentage labels inside the red bars
        for j, group in enumerate(groups_constraints):
            total_val = total_fixed[j]
            change_val = change_dynamic[j]
            percentage_val = (change_val / total_val * 100) if total_val > 0 else 0

            annotations_step.append(dict(
                x=group, y=change_val / 2,  # Position label in the middle of red bar
                text=f"{percentage_val:.1f}%",
                showarrow=False, font=dict(size=12, color="black")
            ))

        # Define the slider step update
        step = dict(
            method="update",
            args=[
                {
                    "y": [total_fixed, change_dynamic],  # Keep total fixed, update red part
                    "base": [None, [0] * len(groups_constraints)]  # Red bars start at zero
                },
                {"title": f"Percentahge: {percentage[i]}%, Objective: {objective_values[i]:.4f}",
                "annotations": annotations_step + [ # Include info box as an annotation
                    dict(
                        x=0.8, y=1.15, xref="paper", yref="paper",
                        text=updated_info_text,
                        showarrow=False,
                        align="right",
                        font=dict(size=12, color="black"),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )]
                    },
            ],
          
            label=f"{percentage[i]}%"  # Slider labels remain
        )
        steps.append(step)

    # Add slider above
    sliders = [dict(
        active=0,
        #currentvalue={"prefix": ""},
        pad={"t": 10},  # Adjust to move the slider higher
        x=0,  #  Keep it closer to the left
        xanchor="left",  # Anchor it properly
        y=1.1,  #  Move it just slightly above the plot (not too high)
        yanchor="top", # Position it above the figure
        steps=steps
    )]

    # # Update layout
    fig.update_layout(
        title=f"Percentage: {percentage[0]}%, Objective: {objective_values[0]:.4f}",
        xaxis_title="Constraint Group",
        yaxis_title="Count",
        sliders=sliders,
        barmode='stack',  # Stack bars like Matplotlib
        annotations=annotations,  # Add initial annotations for percentages
        legend=dict(x=0.9, y=0.95),  # Move legend to the top-right
        margin=dict(t=100, b=50, l=50, r=50)  # Adjust margins to fit slider
    )

    # Save as an interactive HTML file
    html_file = f"{save_path}/{os.path.basename(json_file).replace('.json', '.html')}"
    pio.write_html(fig, html_file)

    print(f"Interactive slider plot saved as '{html_file}'")

if __name__ == '__main__': 
    #main_remove_all_groups(gp.read(open_tepes_9n), 0, 1e-10, 0.1, 'constraints', real_problems)
    # p, obj_values, names_remove = read_json(results_simplified_constraints)
    model = gp.read(open_tepes_9n)
    model = standard_form_e2(model)
    model = normalize_variables(model)
    # #remove_by_group(model, 0, obj_values, names_remove, entity_type = 'constraints')
    # # plot_changes_histogram_with_slider(model)
    importance_constrs, constrs_names = constraints_importance(model)
    group_imp_constr = importances_by_groups(importance_constrs, constrs_names, model, 'constraints')
    plot_changes_histogram_html(model, results_simplified_constraints, 'constraints', group_imp_constr,  real_problems)
    #plot_changes_histogram_html(model, results_simplified_variables, 'variables', real_problems)