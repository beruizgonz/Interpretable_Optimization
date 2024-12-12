import os
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt

from gurobipy import GRB, Model

def pareto_analysis(save_pareto, name, values, values_name, title_x_axis):
    """
    Perform Pareto analysis on the given values and plot a Pareto chart.
    Parameters:
    - save_pareto: Directory to save the Pareto chart.
    - name: Name for the plot and file.
    - values: List of values to analyze.
    - values_name: List of names corresponding to the values.
    - title_x_axis: Title for the x-axis. It is Variable or Constraint.
    """
    # Sort data in descending order of values
    sorted_data = sorted(zip(values, values_name), reverse=True, key=lambda x: x[0])
    values, values_name = zip(*sorted_data)

    # Calculate cumulative sum and percentages
    cumsum_values = np.cumsum(values)
    total_sum = sum(values)
    percentage_values = cumsum_values / total_sum * 100

    # Find the index where cumulative percentage exceeds 80%
    threshold_index = np.argmax(percentage_values >= 80)

    # Plot the Pareto chart
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for values
    ax1.bar(values_name, values, color='skyblue', edgecolor='black', label="Values")
    ax1.set_xlabel(f'{title_x_axis}')
    ax1.set_ylabel('Values')
    ax1.set_title(f'Pareto Analysis {name} ({title_x_axis})')
    #ax1.set_xlim(0, threshold_index)  
      # Modify x-axis labels: Limit to threshold_index or show every nth label
    step = max(1, len(values_name) // 20)  # Show at most 20 labels
    limited_labels = [label if i % step == 0 else '' for i, label in enumerate(values_name)]
    ax1.set_xticks(range(len(values_name)))
    ax1.set_xticklabels(limited_labels, rotation=45, ha='right')

    for i, tick_label in enumerate(limited_labels):
        if tick_label == '':
            ax1.get_xaxis().get_major_ticks()[i].tick1line.set_visible(False)  # Hide the vertical line
            ax1.get_xaxis().get_major_ticks()[i].tick2line.set_visible(False)  # Hide the opposite tick line
            ax1.get_xaxis().get_major_ticks()[i].label1.set_visible(False)  # Hide the corresponding label
  
    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(values_name, percentage_values, color='red', marker='o', label="Cumulative %")
    ax2.set_ylabel('Cumulative Percentage')
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color='green', linestyle='--', label="80% Threshold")
    ax2.axvline(x=threshold_index, color='green', linestyle='--')

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_pareto, f'pareto_{name}.png'))
    plt.close()

def plot_results(vector_epsilon, vector1, vector2, model_name, folder, plot_type="constraints"):
    """
    Generalized function to plot results for constraints or variables.
    
    Parameters:
        vector_epsilon (list/np.array): Epsilon values (thresholds).
        vector1 (list/np.array): Objective values.
        vector2 (list/np.array): Number of constraints/variables.
        model_name (str): Name of the model.
        folder (str): Directory to save the plot.
        plot_type (str): Either "constraints" or "variables" for labeling.
    """
    x_axis = np.array(vector_epsilon)
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Ensure all vectors are of the same length as x_axis by padding with NaN if necessary
    vectors = [vector1, vector2]
    max_length = len(x_axis)
    for i, vec in enumerate(vectors):
        if len(vec) < max_length:
            vectors[i] = np.append(vec, [np.nan] * (max_length - len(vec)))
    vector1, vector2 = vectors

    ### Plot ###
    fig1, axs1 = plt.subplots(2, 1, figsize=(8, 12))
    # reverse the x-axis
    if plot_type == 'variables':
        axs1[0].invert_xaxis()
        init = max(x_axis)
        end = min(x_axis)
    else:
        init = min(x_axis)
        end = max(x_axis)
    # First subplot: Objective Value
    axs1[0].plot(x_axis, vector1, marker='o')
    axs1[0].set_xlim(init,end)  # Adjust as needed
    axs1[0].set_title(f'Objective Value vs. Percentile', fontsize=12)
    axs1[0].set_xlabel('Percentile', fontsize=12)
    axs1[0].set_ylabel('Objective Value', fontsize=12)
    axs1[0].grid(True)

    # Second subplot: Number of Constraints/Variables
    axs1[1].plot(x_axis, vector2, color='orange')
    axs1[1].set_xlim(init, end) # Adjust as needed
    axs1[1].set_title(f'Number of {plot_type.capitalize()} vs. Percentile', fontsize=12)
    axs1[1].set_xlabel('Percentile', fontsize=12)
    axs1[1].set_ylabel(f'Number of {plot_type.capitalize()}', fontsize=12)
    axs1[1].grid(True)

    # Set a title for the entire figure
    fig1.suptitle(f'Results for {model_name} Problem ({plot_type.capitalize()})', fontsize=16)

    # Save the plot
    save_path1 = os.path.join(folder, f'results_{plot_type}_{model_name}.png')
    plt.savefig(save_path1)
    plt.close()
