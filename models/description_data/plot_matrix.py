import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import numpy as np


def plot_sparse_values(matrix):
    # Convert the sparse matrix to coordinate format (COO)
    coo = matrix.tocoo()

    # Normalize the data for better visualization (optional)
    values = coo.data

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        coo.col,         # x-coordinates (columns)
        coo.row,         # y-coordinates (rows)
        c=values,  # Color depends on value
        s=50 * values,  # Size depends on value
        cmap="viridis",  # Color map
        alpha=0.8,       # Transparency for better visibility
        edgecolor="k"    # Edge color for better contrast
    )

    # Add a colorbar to explain the mapping
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Normalized Value")

    # Set axis limits and labels
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(0, matrix.shape[0])
    ax.invert_yaxis()  # Standard matrix order (row 0 at the top)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    plt.title("Sparse Matrix with Values as Color and Size")
    plt.tight_layout()
    plt.show()

def plot_sparse_matrix(matrix):
    # Convert sparse matrix to coordinate format (COO) for visualization
    coo = matrix.tocoo()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coo.col, coo.row, c=coo.data, cmap="Blues", marker="s", s=100)

    # Add annotations for non-zero elements
    for i, j, value in zip(coo.row, coo.col, coo.data):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")

    # Set grid and axis labels
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticks(np.arange(-0.5, matrix.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0]), minor=True)
    ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Adjust axis limits
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(-0.5, matrix.shape[0] - 0.5)

    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.title("Sparse Matrix Visualization")
    plt.tight_layout()
    plt.show()

def plot_structure_matrix(matrix, group_variables):
    # Divide the matrix into blocks based on the group of variables
    blocks_size = []
    for key, value in group_variables.items():
        size_block = value
        blocks_size.append(size_block)

    # Scale the size of the blocks based on the number of variables in each group
    total_size = matrix.shape[1]
    relative_sizes = [size / total_size for size in blocks_size]
    print(relative_sizes)
    # Plot the blocks
    fig, ax = plt.subplots(figsize=(10, 10))
    current_x = 0
    colors = plt.cm.get_cmap("tab10", len(relative_sizes))  # Use a colormap for distinct colors

    for i, (block, size) in enumerate(zip(blocks_size, relative_sizes)):
        ax.add_patch(plt.Rectangle(
            (current_x, 0), matrix.shape[1], matrix.shape[0],
            edgecolor="black", facecolor=colors(i), alpha=0.5
        ))
        current_x += size * total_size

    ax.set_xlim(0, total_size)
    ax.set_ylim(0, matrix.shape[0])
    ax.set_xlabel("Variables")
    ax.set_ylabel("Constraints")
    plt.title("Matrix Structure by Variable Groups")
    plt.tight_layout()
    plt.show()

def plot_changes_histogram(dict_change_groups, dict_group, title):
    """
    Plot a histogram showing the proportion of variables that changed for each group.

    Parameters:
    - dict_change_groups: Dictionary with the number of changes per group.
    - dict_group: Dictionary with the total number of variables per group.
    - title: Title of the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure groups are aligned between the two dictionaries
    groups = list(dict_group.keys())
    total_values = [dict_group[group] for group in groups]
    change_values = [dict_change_groups.get(group, 0) for group in groups]

    # Calculate proportions of changes
    proportions = [changes / total if total > 0 else 0 for changes, total in zip(change_values, total_values)]

    # Create bar positions
    x = np.arange(len(groups))

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, proportions, color="red", alpha=0.7)

    # Set axis labels and title
    ax.set_xlabel("Groups")
    ax.set_ylabel("Proportion of Changes")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def plot_changes_histogram_interactive(EPSILON_NUMBER, model, group_variables, group_constraints_original):
    """
    Interactive plot to visualize changes in variables and constraints by group based on EPSILON_NUMBER.

    Parameters:
    - EPSILON_NUMBER: The epsilon number to filter the results.
    - model: The model to analyze.
    - group_variables: The variable groups in the model.
    - group_constraints_original: The constraint groups in the model.
    """
    # Run the main analysis with the current EPSILON_NUMBER
    changed_group_var, changed_group_constraints = main(model, EPSILON_NUMBER, 'Sparsification')
    count_by_group_var = analyzes_groups(changed_group_var)
    count_by_group_constraints = analyzes_groups(changed_group_constraints)
    total_variances = get_total_by_group(group_variables)
    total_constraints = get_total_by_group(group_constraints_original)

    # Plot changes in variables by group
    def plot_histogram(dict_change_groups, dict_group, title):
        """
        Plot a histogram showing the proportion of variables/constraints that changed.
        """
        groups = list(dict_group.keys())
        total_values = [dict_group[group] for group in groups]
        change_values = [dict_change_groups.get(group, 0) for group in groups]

        proportions = [changes / total if total > 0 else 0 for changes, total in zip(change_values, total_values)]
        x = np.arange(len(groups))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, proportions, color="red", alpha=0.7)
        ax.set_xlabel("Groups")
        ax.set_ylabel("Proportion of Changes")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot
        filename = f"{title.replace(' ', '_')}_epsilon_{EPSILON_NUMBER}.png"
        plt.savefig(filename)
        plt.show()
        print(f"Plot saved as {filename}")

    # Generate plots
    plot_histogram(count_by_group_var, total_variances, 'Changes in Variables by Group')
    plot_histogram(count_by_group_constraints, total_constraints, 'Changes in Constraints by Group')


def plot_changes_histogram(dict_change_groups, dict_group, title, epsilon_number):
    """
    Plot a histogram showing the proportion of variables/constraints that changed.

    Parameters:
    - dict_change_groups: Dictionary with the number of changes per group.
    - dict_group: Dictionary with the total number of variables per group.
    - title: Title of the plot.
    - epsilon_number: The epsilon number used for the plot (for saving the file).
    """
    groups = list(dict_group.keys())
    total_values = [dict_group[group] for group in groups]
    change_values = [dict_change_groups.get(group, 0) for group in groups]

    proportions = [changes / total if total > 0 else 0 for changes, total in zip(change_values, total_values)]
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, proportions, color="red", alpha=0.7)
    ax.set_xlabel("Groups")
    ax.set_ylabel("Proportion of Changes")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    filename = f"{title.replace(' ', '_')}_epsilon_{epsilon_number}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved as {filename}")


def interactive_plot_loop(model, group_variables, group_constraints_original):
    """
    Interactive loop to allow dynamic selection of epsilon number and plot updates.

    Parameters:
    - model: The model to analyze.
    - group_variables: The variable groups in the model.
    - group_constraints_original: The constraint groups in the model.
    """
    while True:
        try:
            epsilon_number = int(input("Enter the EPSILON_NUMBER (or -1 to exit): "))
            if epsilon_number == -1:
                print("Exiting interactive plot loop.")
                break

            # Run analysis with the current EPSILON_NUMBER
            changed_group_var, changed_group_constraints = main(model, epsilon_number, 'Sparsification')
            count_by_group_var = analyzes_groups(changed_group_var)
            count_by_group_constraints = analyzes_groups(changed_group_constraints)
            total_variances = get_total_by_group(group_variables)
            total_constraints = get_total_by_group(group_constraints_original)

            # Generate plots
            plot_changes_histogram(count_by_group_var, total_variances, 'Changes in Variables by Group', epsilon_number)
            plot_changes_histogram(count_by_group_constraints, total_constraints, 'Changes in Constraints by Group', epsilon_number)

        except ValueError:
            print("Invalid input. Please enter a valid integer or -1 to exit.")