import gurobipy as gp
import numpy as np
import math
from scipy.sparse import csr_matrix
import sys
import os 
import openpyxl
import pandas as pd
import logging 

from datetime import datetime
from utils_models.utils_functions import * 
from utils_models.standard_model import *
from new_main import sensitivity_analysis, load_class

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

# ACCESS TO THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'INDUS.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'GUSSGRID.mps')
results_folder = os.path.join(project_root, 'results')
save_path = os.path.join(results_folder, 'sparsification_optimal_solution')

if not os.path.exists(save_path):
    os.makedirs(save_path)

def set_new_bounds(file, alpha_min, alpha_max):
    """
    Set new bounds for the variables in the model based on the optimal solution.
    
    Parameters:
        model (gurobipy.Model): The optimization model to update.
        alpha_min (float): Scaling factor for the lower bound (multiplier of the optimal value).
        alpha_max (float): Scaling factor for the upper bound (multiplier of the optimal value).
    """
    # Open the model
    model = gp.read(file)
    try:
        # Solve the model to find the optimal solution
        model = standard_form_e1(model)
        model.setParam('OutputFlag', 0)
        model.optimize()

        if model.status != gp.GRB.OPTIMAL:
            print("No optimal solution found. Bounds not updated.")
            return

        print("Optimal solution found. Updating variable bounds.")

        # Update bounds for each variable based on the optimal solution
        for var in model.getVars():
            optimal_value = var.x  # Get the optimal value
            if optimal_value is not None:
                # Set new bounds
                var.lb = optimal_value * alpha_min
                var.ub = optimal_value * alpha_max

        # Update the model to apply new bounds
        model.update()
        print("Variable bounds updated successfully.")

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e.message}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    # optimize the model
    model.optimize()
    # Print the optimal objective value
    print(f"Optimal objective value: {model.objVal}")
    # Comprobate the new bounds, see that all are finite
    lower_bounds = [var.lb for var in model.getVars()]
    upper_bounds = [var.ub for var in model.getVars()]
    bool_lower = all([np.isfinite(lb) for lb in lower_bounds])
    bool_upper = all([np.isfinite(ub) for ub in upper_bounds])
    return model.objVal, bool_lower, bool_upper

def sparsification_bounds(file, save_path, alpha_min = 0.0001, alpha_max= 1000):
    results = nested_dict()
    model = gp.read(file)
    try:
    # Solve the model to find the optimal solution
        model = standard_form(model)
        model.setParam('OutputFlag', 0)
        model.optimize()

        if model.status != gp.GRB.OPTIMAL:
            print("No optimal solution found. Bounds not updated.")
            return

        print("Optimal solution found. Updating variable bounds.")

        # Update bounds for each variable based on the optimal solution
        for var in model.getVars():
            optimal_value = var.x  # Get the optimal value
            if optimal_value is not None:
                # Set new bounds
                var.lb = optimal_value
                var.ub = optimal_value

        # Update the model to apply new bounds
        model.update()
        print("Variable bounds updated successfully.")

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e.message}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    model_to_use_dual = construct_dual_model(model)
    print_model_in_mathematical_format(model)
    presolve = load_class(model)
    start_time = time.time()
    results_problem =  sensitivity_analysis(model, presolve)
    model_name = file.split('.')[0]
    model_name = model_name.split('/')[-1]
    results[model_name]['primal'] = results_problem
    results[model_name]['primal']['time_required'] = time.time() - start_time
    results[model_name]['primal']['mathematical-model'] = get_model_in_mathematical_format(model)

    log.info(
                f"{str(datetime.now())}:Sensitivity Analysis with dual model..."
            )

    start_time = time.time()
    presolve_dual = load_class(model_to_use_dual)

    # Set the output flag to 0
    model_to_use_dual.setParam('OutputFlag', 0)
    model_to_use_dual.optimize()
    # Get the variables
    dv_dual = np.array([var.x for var in model_to_use_dual.getVars()])
    of = model_to_use_dual.objVal
    results_problem_dual = {'objective_function': of,
                            'decision_variables': dv_dual}
    results[model_name]['dual'] = results_problem_dual
    results[model_name]['dual']['time_required'] = time.time() - start_time
    results[model_name]['dual']['mathematical_model'] = get_model_in_mathematical_format(
        model_to_use_dual)
    
    name = file.split('/')[-1]
    name = name.split('.')[0]
    model_save_path = os.path.join(save_path, f'epsilon_sparsification_{name}.json')
    dict2json(results, model_save_path)
    return results

def verify_bound(folder):
    wb = openpyxl.Workbook()
    wb_excel = wb.active
    wb_excel.title = 'Results'
    data = pd.DataFrame()
    # Create three columns to store the results
    wb_excel['A1'] = 'Model'
    wb_excel['B1'] = 'Objective'
    wb_excel['C1'] = 'Objective_bounds'
    wb_excel['D1'] = 'Lower_bounds'
    wb_excel['E1'] = 'Upper_bounds'
    for model_name in os.listdir(GAMS_path_modified):
        if model_name.endswith('.xlsx'):
            continue
        model_path = os.path.join(GAMS_path, model_name)
        model = gp.read(model_path)
        model.setParam('OutputFlag', 0)
        model.optimize()
        obj = model.objVal
        objective, bool_lower, bool_upper = set_new_bounds(model, 0.0001, 10000)
        # In a row put the model name, the objective value and the objective value with the new bounds
        wb_excel.append([model_name, obj, objective, bool_lower, bool_upper])
    # Save the results in a excel file
    wb.save('results.xlsx')

if __name__ == '__main__':
    # Set new bounds for the variables in the model
    # set_new_bounds(model_path, 0.0001, 1000)
    # Verify that the new bounds are finite
    # verify_bound(GAMS_path)
    for name in os.listdir(GAMS_path_modified):
        if name == 'INDUS89.mps':
            continue
        model_path = os.path.join(GAMS_path_modified, f'{name}')
        results = sparsification_bounds(model_path,save_path = save_path, alpha_min = 0.0001, alpha_max = 1000)
    #sparsification_bounds(model_path_modified, save_path = save_path, alpha_min = 0.0, alpha_max = 0)