import json
import time
import copy
import os
import gurobipy as gp
import logging
import sys
from tabulate import tabulate
from datetime import datetime
import traceback
import math
import pandas as pd

from opts import parse_opts
from utils_models.presolvepsilon_class import PresolvepsilonOperations
from utils_models.sensitivity_analysis import SensitivityAnalysis
from utils_models.utils_functions import *

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

# Load the data information
opts = parse_opts()
opts.eliminate_zero_rows_epsilon = True


#The path to the data
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'ROBERT_modified.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'INDUS89.mps')

def load_class(model):
    presolve = PresolvepsilonOperations(model, eliminate_zero_rows_epsilon = True, opts = opts)
    return presolve

# presolve.load_model_matrices()
# #presolve.sparsification_operation()
# presolve.eliminate_zero_rows_operation(epsilon = 0.0015)


def build_normalized_model(A_norm, b_norm, c, lb, ub, of_sense, cons_senses, var_names):
    """
    Builds a Gurobi model using the normalized A and b, and original c, lb, ub.
    """
    model = gp.Model()

    # Set objective sense
    model.ModelSense = of_sense  # 1 for minimization, -1 for maximization

    # Add variables
    num_vars = len(c)
    variables = []
    for i in range(num_vars):
        var = model.addVar(lb=lb[i], ub=ub[i], obj=c[i], name=var_names[i])
        variables.append(var)
    model.update()

    # Add constraints
    num_constrs = len(b_norm)
    for i in range(num_constrs):
        expr = gp.LinExpr()
        for j in range(num_vars):
            if A_norm[i, j] != 0:
                expr.add(variables[j], A_norm[i, j])
        sense = cons_senses[i]
        rhs = b_norm[i]
        if sense == '<':
            model.addConstr(expr <= rhs)
        elif sense == '>':
            model.addConstr(expr >= rhs)
        elif sense == '=':
            model.addConstr(expr == rhs)
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")
    model.update()

    return model



def sensitivity_analysis(model, presolve, min_threshold = 0.0015, max_threshold = 0.25, step = 0.3):

    original_model = model.copy()
    model.setParam('OutputFlag', 0)
    model.optimize()
    eps = [0]  # Start with 0 threshold
    execution_time = [0]
    dv = [np.array([var.x for var in model.getVars()])]  # Start with decision variables of original model
    changed_indices = [None]  # List to store indices changed at each threshold
    abs_vio, obj_val = measuring_constraint_infeasibility(model, dv[0])
    of = [obj_val]  # Start with the objective value of the original model
    constraint_viol = [abs_vio]  # List to store infeasibility results
    of_dec = [model.objVal]
    iteration = 1  # Initialize iteration counter
    original_model = model.copy()

        # A_initial = A.A.copy()
        # #print(A_initial.shape)
        # original_model = model.copy()

        # Calculate total iterations for exponential growth
    total_iterations = math.ceil(
        math.log(
            max_threshold /
            min_threshold

        ) / math.log(
            1 + step
        )
    )

    (A, b, c, lb, ub, of_sense, cons_senses, co, variable_names)   = get_model_matrices(model = original_model)
    # model = build_normalized_model(A, b, c, lb, ub, of_sense, cons_senses, variable_names)
    # model.optimize()
    # if model.Status == GRB.OPTIMAL:
    #     print('Optimal objective value (normalized model):', model.ObjVal)
    #     solution = model.getAttr('X', model.getVars())
    # # for var_name, value in zip(variables_names, solution):
    # #     print(f'{var_name}: {value}')
    # else:
    #     print('Optimization was not successful.')

    # copy initial A matrix and if then modify it don't modify the original
    A_initial = A.copy()
    threshold = min_threshold
    while threshold <= max_threshold:
        start_time = datetime.now()
        (A_new, b, c, lb, ub, of_sense, cons_senses, co, variable_names,
                change_dictionary, operation_table)   = presolve.orchestrator_presolve_operations(model = original_model, epsilon = threshold)
        A_changed = A_new.copy()
        #print(np.all(of_sense == cons_senses_initial))

        indices = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1]) if
                       A_initial[i, j] != 0 and A_changed[i, j] == 0]
        # Delete the row with all zeros
        if len(indices) > 0:
            print('Cambia')
        save_json(A_changed, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
        iterative_model = build_model_from_json(opts.save_path)

        # Solve the new model
        iterative_model.setParam('OutputFlag', 0)
        iterative_model.optimize()
        # compare_models(original_model, iterative_model)
        # Update lists with results
        # Check if the model found a solution
        if iterative_model.status == gp.GRB.OPTIMAL:
            # Model found a solution
            eps.append(threshold)
            of.append(iterative_model.objVal)
            changed_indices.append(indices)
            dv.append(np.array([var.x for var in iterative_model.getVars()]))
            # Access the decision variables
            variables = iterative_model.getVars()
            # Extract their values and store in a NumPy array
            decisions = np.array([var.x for var in variables])
            # calculate the constraint infeasibility
            abs_vio, obj_val = measuring_constraint_infeasibility(original_model, decisions)

            # Store infeasibility results
            of_dec.append(obj_val)
            constraint_viol.append(abs_vio)
        else:
            # Model did not find a solution
            eps.append(threshold)
            of.append(np.nan)  # Append NaN for objective function
            of_dec.append(np.nan)
            changed_indices.append(np.nan)
            constraint_viol.append(np.nan)
            dv.append(np.full(len(original_model.getVars()), np.nan))

        # Delete the model to free up resources
        del iterative_model

        # Progress bar
        progress = iteration / total_iterations
        bar_length = 50  # Length of the progress bar
        progress_bar = '>' * int(bar_length * progress) + ' ' * (bar_length - int(bar_length * progress))
        sys.stdout.write(f"\r[{progress_bar}] Iteration {iteration}/{total_iterations}")
        sys.stdout.flush()

        iteration += 1  # Increment iteration counter

        # Increment threshold
        threshold += step*threshold

    #     execution_time.append(datetime.now() - start_time)
    results = {
        'epsilon': eps,
        'objective_function': of,
        'decision_variables': dv,
        'changed_indices': changed_indices,
        'constraint_violation': constraint_viol,
        'of_original_decision': of_dec,
        'execution_time': execution_time,
    }
    return results

def standar():
    GAMS_path = os.path.join(project_root, 'data/GAMS_library')
    print(len(os.listdir(GAMS_path)))
    for model_name in os.listdir(GAMS_path):
        if model_name.endswith('.xlsx'):
            continue
        model_path = os.path.join(GAMS_path, model_name)
        model = gp.read(model_path)
        # Solve the problem in canonical form and standard form
        original_primal, track_elements = canonical_form(model)
        standard_model = standard_form1(model)
        print_model(original_primal)
        print_model(standard_model)
        original_primal.setParam('OutputFlag', 0)
        standard_model.setParam('OutputFlag', 0)
        original_primal.optimize()
        standard_model.optimize()
            # Check if the objective values are the same
        #if original_primal.Status == GRB.OPTIMAL and standard_model.Status == GRB.OPTIMAL:
        print(model_name[-1], original_primal.objVal,  standard_model.objVal)

def sensitivity_analysis_file(file):
    results = nested_dict()
    # for model_name in os.listdir(folder):
    #     if model_name.endswith('.xlsx') or model_name.endswith('modified.mps') or model_name == 'AJAX.mps':
    #         continue
    #     model_path = os.path.join(folder, model_name)
    model = gp.read(file)
    #model = gp.read(model_path)
    original_primal, track_elements = canonical_form(model,minOption=True)
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
    save_json(A, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
    # Load the dual model
    model_to_use_dual = build_dual_model_from_json(opts.save_path)

    presolve = load_class(original_primal)
    start_time = time.time()
    results_problem =  sensitivity_analysis(original_primal, presolve)
    model_name = file.split('.')[0]
    model_name = model_name.split('/')[-1]
    results[model_name]['primal'] = results_problem
    results[model_name]['primal']['time_required'] = time.time() - start_time
    results[model_name]['primal']['mathematical-model'] = get_model_in_mathematical_format(original_primal)

    log.info(
                f"{str(datetime.now())}:Sensitivity Analysis with dual model..."
            )

    start_time = time.time()
    presolve_dual = load_class(model_to_use_dual)
    results_problem_dual = sensitivity_analysis(model_to_use_dual, presolve_dual)
    results[model_name]['dual'] = results_problem_dual
    results[model_name]['dual']['time_required'] = time.time() - start_time
    results[model_name]['dual']['mathematical_model'] = get_model_in_mathematical_format(
        model_to_use_dual)

    dict2json(results, 'epsilon_results.json')

def sensitivity_analysis_folder(folder):
    results = nested_dict()
    for model_name in os.listdir(folder):
        if model_name.endswith('.xlsx') or model_name.endswith('modified.mps') or model_name == 'AJAX.mps':
            continue
        model_path = os.path.join(folder, model_name)
        model = gp.read(model_path)
        original_primal, track_elements = canonical_form(model,minOption=True)
        A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
        save_json(A, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
        # Load the dual model
        model_to_use_dual = build_dual_model_from_json(opts.save_path)

        presolve = load_class(original_primal)
        start_time = time.time()
        results_problem =  sensitivity_analysis(original_primal, presolve)
        model_name = model_name.split('.')[0]
        results[model_name]['primal'] = results_problem
        results[model_name]['primal']['time_required'] = time.time() - start_time
        results[model_name]['primal']['mathematical-model'] = get_model_in_mathematical_format(original_primal)

        log.info(
                    f"{str(datetime.now())}:Sensitivity Analysis with dual model..."
                )

        start_time = time.time()
        presolve_dual = load_class(model_to_use_dual)
        results_problem_dual = sensitivity_analysis(model_to_use_dual, presolve_dual)
        results[model_name]['dual'] = results_problem_dual
        results[model_name]['dual']['time_required'] = time.time() - start_time
        results[model_name]['dual']['mathematical_model'] = get_model_in_mathematical_format(
            model_to_use_dual)

    dict2json(results, 'epsilon_results.json')

def main_bounds(folder):
    # Create a pandas dataframe
    rows = []
    objective_bounds = 0  # Initialize objective_bounds to 0

    for model_name in os.listdir(folder):
        # Skip Excel files if present in the folder
        if model_name.endswith('.xlsx'):
            continue

        # Construct the full path of the model
        model_path = os.path.join(folder, model_name)
        name = model_name.split('.')[0]  # Extract the base name of the model

        try:
            # Load the model
            model = gp.read(model_path)  # Assuming the model is in a valid format (like MPS or LP)
            standard_model = standard_form1(model)

            # Uncomment these if you want to calculate bounds later
            standard_model.setParam('OutputFlag', 0)
            standard_model.optimize()
            # lb, ub = calculate_bounds2(standard_model)
            # bounds_model = build_model_from_json('model_matrices.json')
            # bounds_model.setParam('OutputFlag', 0)
            # bounds_model.optimize()
            # if bounds_model.Status == gp.GRB.OPTIMAL:
            #     objective_bounds = bounds_model.objVal

            # Check if the optimization status is optimal
            if standard_model.Status == gp.GRB.OPTIMAL:
                objective_standard = standard_model.objVal
                print(f'Objective value without bounds: {objective_standard}')
            else:
                objective_standard = 0  # Set to 0 if not optimal

        except gp.GurobiError as e:
            print(f"Error reading or optimizing model {model_name}: {e}")
            objective_standard = 0  # If there's an error, set the objective to 0

        # Add a new row with model name, standard objective, and bounds objective
        new_row = [name, objective_standard, objective_bounds]
        rows.append(new_row)

    # Create a DataFrame with the collected rows
    results = pd.DataFrame(rows, columns=['Model', 'Standard', 'Bounds'])
    #order the results by alphabetical order of the model name
    results = results.sort_values(by='Model')
    # Save the DataFrame to an Excel file
    try:
        results.to_excel('results_bounds.xlsx', index=False)
        print("Results successfully saved to 'results_bounds.xlsx'")
    except Exception as e:
        print(f"Failed to save results to Excel: {e}")



if __name__ == '__main__':
    # # results = nested_dict()
    # # for model_name in os.listdir(GAMS_path):
    # #     model_path = os.path.join(GAMS_path, model_name)
    # #     if model_name.endswith('.xlsx'):
    # #         continue
    # #     model = gp.read(model_path)
    # #     original_primal, track_elements = canonical_form(model)

    # #     presolve = load_class(original_primal)
    # #     results_problem =  sensitivity_analysis(original_primal, presolve)
    # #     results[model_name] = results_problem

    # # dict2json(results, 'epsilon_results.json')

    """
    ------------------------CALCULATE THE BOUNDS OF THE MODEL------------------------
    """
    model_name = model_path_modified
    model = gp.read(model_name)
    model.optimize()
    print(model.objVal)
    # Get the model in standard form
    #A, b, obj = parse_mps(model_name)
    standard_model = standard_form1(model)
    # canonical_model, _ = canonical_form(model)
    # canonical_model.setParam('OutputFlag', 0)
    standard_model.optimize()
    # #print_model_in_mathematical_format(standard_model)

    print(standard_model.objVal)
    # # for var in standard_model.getVars():
    #     print(f'{var.VarName}: {var.X}')
    # #print_model_in_mathematical_format(standard_model)
    # lb, ub = calculate_bounds2(standard_model)
    # print(f'Lower bounds: {lb}')
    # print(f'Upper bounds: {ub}')
    # # # See if lb <= ub
    # # print(all(lb_new <= ub_new))
    # path = os.path.join(os.getcwd(), 'model_matrices.json')
    # model = build_model_from_json(path)
    # print_model_in_mathematical_format(model)
    # model.setParam('OutputFlag', 0)
    # model.optimize()
    # print(model.objVal)
    # for var in model.getVars():
    #     print(f'{var.VarName}: {var.X}')
    # # Compute the sensitivity analysis

    #main_bounds(GAMS_path_modified)

    """
    ----------------SENSITIVITY ANALYSIS FOR A SINGLE MODEL----------------
    """

    # sensitivity_analysis_file(model_path)
    # model_name = os.path.join(GAMS_path, 'SHALE.mps')
    # model = gp.read(model_name)
    # original_primal, track_elements = canonical_form(model)
    # presolve = load_class(original_primal)
    # results_problem =  sensitivity_analysis(original_primal, presolve)
    # dict2json(results_problem, 'epsilon_results.json')
    # print(results_problem)
    # sensitivity_analysis_folder(GAMS_path)

