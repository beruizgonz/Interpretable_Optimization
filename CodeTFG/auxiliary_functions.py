from matplotlib import pyplot as plt
import numpy as np
import os
# Function to calculate the media of a list.
def calculate_means(lst):
    means = []
    for sublist in lst:
        if isinstance(sublist, list):
            if len(sublist) > 1:
                mean = np.nanmean(sublist)
                means.append(mean)
            elif len(sublist) == 1:
                means.append(sublist[0])
            else:
                means.append(None)
        elif isinstance(sublist, (float, int)):
            means.append(sublist)
    return means

def calculate_lengths(vector_of_vectors):
    # Create a list to store the lengths of the vectors
    lengths = []
    # Iterate over each vector in the vector_of_vectors
    for vector in vector_of_vectors:
        # If the vector is None, consider its length as 0
        if vector is None:
            lengths.append(0)
        # If the vector contains NaN, exclude it from the length calculation
        elif isinstance(vector, (list, np.ndarray)) and np.isnan(vector).any():
            continue
        else:
            # Check if the vector is iterable before attempting to calculate its length
            try:
                length = len(vector)
            except TypeError:
                # If it is not iterable, append 0 to the lengths
                length = 0
            lengths.append(length)
    return lengths

def set_values_below_threshold_to_zero(list_of_lists, reference_value):
    # Iterate over each sublist in the list of lists. It removes values smaller than the reference value, as if there were no violation.
    for i in range(len(list_of_lists)):
        for j in range(len(list_of_lists[i])):
            if abs(list_of_lists[i][j]) < reference_value:
                list_of_lists[i][j] = 0
    return list_of_lists

# FOR SOME MODELS LIKE AIRCRAFT, NaN SUBLISTS APPEAR AND NEED TO BE REMOVED
def remove_nan_sublists(list_of_lists):
    list_without_nan = [sublist for sublist in list_of_lists if not np.any(np.isnan(sublist))]
    return list_without_nan

def multiply_matrices(A, B):
    # Get the dimensions of the matrices
    rows_A = len(A)
    columns_A = len(A[0])
    rows_B = len(B)
    columns_B = len(B[0])
    
    # Check if the matrices are compatible for multiplication
    if columns_A != columns_B:
        raise ValueError("The matrices cannot be multiplied")
    
    # Initialize the resulting matrix with zeros
    C = [[0] * columns_B for _ in range(rows_A)]
    
    # Multiply element by element and sum the results
    for i in range(rows_A):
        for j in range(columns_B):
            for k in range(columns_A):
                C[i][j] = A[i][j] * B[i][j]
    return C

def sum_sublists(list_of_lists):
    return [sum(map(float, sublist)) for sublist in list_of_lists]

def plot1(model_pr_epsilon, vector1, title, name1):
    # Determine the maximum length of the vectors
    max_length = max(len(model_pr_epsilon), len(vector1))
    
    # Generate the x-axis with the same length as the longest vector
    x_axis = model_pr_epsilon[:max_length]
    
    # Plot the vectors
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis[:len(vector1)], vector1, label=name1)
    
    # Set labels and title with font sizes
    plt.xlabel('Epsilon', fontsize=15)
    plt.ylabel('', fontsize=12)
    plt.title(title, fontsize=16)
    
    # Set legend with font size
    plt.legend(fontsize=12)
    
    # Set tick label size
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Set other graph parameters
    plt.grid(False)
    plt.xlim(x_axis[0], x_axis[-1]) # Adjust x-axis limits
    # Show the graph
    plt.show()


def plot_subplots(folder, model_name, vector_epsilon, vector1, vector2, vector3, name1, name2, name3):
    # Determine the maximum length of the vectors
    max_length = max(len(vector_epsilon), len(vector1), len(vector2), len(vector3[0]))

    # Generate the x-axis with the same length as the longest vector
    total, rows, cols = vector3[0], vector3[1], vector3[2]
    x_axis = vector_epsilon[:max_length]
    
    # Plot with subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 18))

    # For the first subplot I want the y-axis to be 0 to 100
    axs[0].plot(x_axis[:len(vector1)], vector1)
    axs[1].plot(x_axis[:len(vector2)], vector2)
    axs[2].plot(x_axis[:len(total)], total)
    axs[2].plot(x_axis[:len(rows)], rows)   
    axs[2].plot(x_axis[:len(cols)], cols)

    # Set labels and title with font sizes
    axs[0].set_title(name1, fontsize=12)
    axs[1].set_title(name2, fontsize=12)
    axs[2].set_title(name3, fontsize=12)

    # ax[2] add legend
    axs[2].legend(['Total', 'Rows', 'Columns'])

        # Set a title for the entire plot
    fig.suptitle(f'Sparsification indexes for {model_name} problem', fontsize=16)
    save_path = os.path.join(folder, f'sparsification_indexes_{model_name}.png')
    plt.savefig(save_path)
    #plt.show()
    plt.close()



def convert_late_zeros_to_nan(vector):
    found_non_zero = False
    for i, num in enumerate(vector):
        if num != 0:
            found_non_zero = True
        elif found_non_zero and num == 0:
            vector[i] = float('nan')
    return vector