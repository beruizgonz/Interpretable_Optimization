import os
import gurobipy as gp
import sys
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from models.opts import parse_opts
from models.utils_models.presolvepsilon_class import PresolvepsilonOperations
from models.utils_models.utils_functions import *
from models.utils_models.standard_model import standard_form, construct_dual_model, standard_form2

# Path to the data folder
parent_dir = os.path.dirname(os.getcwd())
GAMS_models_folder = os.path.join(parent_dir, 'data/GAMS_library_modified')


def bounds(model_name):
    """
    Calculate the bounds for a given model.

    Parameters:
        model (str): The model file path.

    Returns:
        tuple: A tuple containing lower bounds (lb) and upper bounds (ub).
    """
    model = gp.read(model_name)
    standard_model = standard_form(model)  # Assuming `standard_form` is properly defined
    lb, ub = calculate_bounds(standard_model)  # Assuming `calculate_bounds2` returns bounds
    return lb, ub


def excel_bounds(path_excel, models):
    """
    Save the bounds information in an Excel file.

    Parameters:
        path_excel (str): The path where the Excel file will be saved.
        models (list): A list of models to evaluate the bounds for.

    Raises:
        ValueError: If any of the upper bounds for the models are not finite.
    """
    # Initialize the dictionary to collect bounds status
    data = {}

    # Iterate over each model and check bounds
    for model in models:
        model_name = os.path.basename(model)
        lb, ub = bounds(model)
        # Check if all upper bounds are finite
        if not np.all(np.isfinite(ub)):
            data[model_name] = 'Infinite'
        else:
            data[model_name] = 'Finite'
            print(f"Model {model} has finite upper bounds")

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Upper Bounds'])

    # Save the DataFrame to an Excel file
    df.to_excel(path_excel, index_label='Model')

# Define a simple problem to see the bounds stretching
def simple_problem(): 
    m = gp.Model("simple")
    x = m.addVar(lb=0, ub=gp.GRB.INFINITY, name="x")
    y = m.addVar(lb=0, ub=gp.GRB.INFINITY, name="y")
    m.setObjective(x + y, gp.GRB.MINIMIZE)
    m.addConstr(x + y <= 3)
    m.addConstr(x - y <= 5)
    m.optimize()
    return m


if __name__ == '__main__':
    # # Ensure the GAMS models folder is correctly loaded and iterate over models
    # models = []

    # # Collecting model paths from the GAMS models folder
    # for model_name in os.listdir(GAMS_models_folder):
    #     model_path = os.path.join(GAMS_models_folder, model_name)
    #     models.append(model_path)
    #     print(f"Loaded model: {model_path}")

    # # Define the path to save the Excel file
    # output_excel_path = os.path.join(parent_dir, 'bounds_output.xlsx')

    # # Save the bounds information for all models
    # try:
    #     excel_bounds(output_excel_path, models)
    #     print(f"Bounds information saved successfully to {output_excel_path}")
    # except ValueError as e:
    #     print(f"Error: {e}")
    model = simple_problem()
    print_model_in_mathematical_format(model)
    standar_model = standard_form(model)
    print_model_in_mathematical_format(standar_model)
    lb, ub = calculate_bounds(standar_model)    
    print(lb, ub)
