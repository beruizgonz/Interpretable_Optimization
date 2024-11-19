import json
import time
import os
import gurobipy as gp
import logging
import sys
from tabulate import tabulate
from datetime import datetime
import pickle
import math
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import gc

from opts import parse_opts
from utils_models.presolvepsilon_class import PresolvepsilonOperations
from utils_models.utils_functions import *
from utils_models.standard_model import standard_form1, construct_dual_model, standard_form

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

# Load the data information
opts = parse_opts()
opts.eliminate_zero_rows_epsilon = True


#The path to the data
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'INDUS.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'DINAM.mps')

def load_class(model):
    presolve = PresolvepsilonOperations(model, eliminate_zero_rows_epsilon = True, opts = opts)
    return presolve

def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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


def sensitivity_analysis_new(model, presolve, dual=False, min_threshold=0.015, max_threshold=0.3, step=0.2):
    model.setParam('OutputFlag', 0)
    model.optimize()
    eps = [0]
    execution_time = [0]
    print('Optimal solution found: %s' % (model.objVal))
    dv = [np.array([var.x for var in model.getVars()])]
    changed_indices = [None]
    rows_changed = [None]
    columns_changed = [None]
    abs_vio, obj_val = measuring_constraint_infeasibility(model, dv[0])
    of = [obj_val]
    constraint_viol = [abs_vio]
    of_dec = [model.objVal]
    iteration = 1
    original_model = model.copy()

    (A, b, c, lb, ub, of_sense, cons_senses, co, variable_names) = get_model_matrices(model=original_model)
    A_initial = csr_matrix(A)  # Use sparse matrix
    non_zeros = A_initial.nonzero()
    total_non_zeros = len(non_zeros[0])
    threshold = min_threshold

    # Calculate total iterations for progress bar
    total_iterations = math.ceil(math.log(max_threshold / min_threshold) / math.log(1 + step))

    while threshold <= max_threshold:
        print(f"Threshold: {threshold}")
        (A_new, b, c, lb, ub, of_sense, cons_senses, co, variable_names,
         change_dictionary, operation_table) = presolve.orchestrator_presolve_operations(
            model=original_model, epsilon=threshold
        )
        non_zeros = A_initial.nonzero()
        total_non_zeros = len(non_zeros[0])
        print(total_non_zeros)
        A_changed = csr_matrix(A_new)  # Use sparse matrix
        indices = [(int(i), int(j)) for i, j in zip(*non_zeros) if A_changed[i, j] == 0]
        rows_zero = np.where(~A_changed.toarray().any(axis=1))[0]
        columns_zero = np.where(~A_changed.toarray().any(axis=0))[0]

        if len(rows_zero) > 0 or len(indices) > 0:
            print('changed')

        save_json(A_changed, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)

        iterative_model = (construct_dual_model_from_json if dual else construct_model_from_json)(opts.save_path)
        start_time = datetime.now()
        iterative_model.setParam('OutputFlag', 0)
        iterative_model.optimize()
        execution_time.append(datetime.now() - start_time)

        if iterative_model.status == gp.GRB.OPTIMAL:
            eps.append(threshold)
            of.append(iterative_model.objVal)
            changed_indices.append(indices)
            rows_changed.append(rows_zero)
            columns_changed.append(columns_zero)
            dv.append(np.array([var.x for var in iterative_model.getVars()]))
            variables = iterative_model.getVars()
            decisions = np.array([var.x for var in variables])
            abs_vio, obj_val = measuring_constraint_infeasibility(original_model, decisions)
            of_dec.append(obj_val)
            constraint_viol.append(abs_vio)
        else:
            eps.append(threshold)
            of.append(np.nan)
            of_dec.append(np.nan)
            changed_indices.append(indices)
            rows_changed.append(rows_zero)
            columns_changed.append(columns_zero)
            constraint_viol.append(np.nan)
            dv.append(np.full(len(original_model.getVars()), np.nan))

        del iterative_model
        #gc.collect()  # Free up memory

        # Progress bar
        progress = iteration / total_iterations
        sys.stdout.write(f"\r[{'>' * int(50 * progress):<50}] Iteration {iteration}/{total_iterations}")
        sys.stdout.flush()

        iteration += 1
        threshold += step * threshold

    results = {
        'epsilon': eps,
        'objective_function': of,
        'decision_variables': dv,
        'changed_indices': changed_indices,
        'rows_changed': rows_changed,
        'columns_changed': columns_changed,
        'constraint_violation': constraint_viol,
        'of_original_decision': of_dec,
        'non_zeros': total_non_zeros,
        'execution_time': execution_time,
    }

    return results

def sensitivity_analysis(model, presolve,  dual = False, min_threshold = 0.015, max_threshold = 0.3, step = 0.2):

    #original_model = model.copy()
    model.setParam('OutputFlag', 0)
    model.optimize()
    eps = [0]  # Start with 0 threshold
    execution_time = [0]
    dv = [np.array([var.x for var in model.getVars()])]  # Start with decision variables of original model
  
    changed_indices = [None]  # List to store indices changed at each threshold
    rows_changed = [None]
    columns_changed = [None]
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
    A_initial = A.copy()
    # Save the non zero elements of the model
    non_zeros = A_initial.nonzero()
    # Save the non zero elements of the mode the total number of non zero elements
    total_non_zeros = len(non_zeros[0])
    threshold = min_threshold
    while threshold <= max_threshold:
        
        print(f"Threshold: {threshold}")
        (A_new, b, c, lb, ub, of_sense, cons_senses, co, variable_names,
                change_dictionary, operation_table)   = presolve.orchestrator_presolve_operations(model = original_model, epsilon = threshold)
        A_changed = A_new.copy()
        #print(np.all(of_sense == cons_senses_initial))

        indices = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1]) if
                       A_initial[i, j] != 0 and A_changed[i, j] == 0]
        A_changed1 = A_changed.A.copy()
        rows_zero = np.where(~A_changed1.any(axis=1))[0]
        columns_zero = np.where(~A_changed1.any(axis=0))[0]
        if len(rows_zero) > 0:
            print('changed')
        save_json(A_changed, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
        if not dual:
             iterative_model = construct_model_from_json(opts.save_path)
        else:
            iterative_model = construct_dual_model_from_json(opts.save_path)
        # Solve the new model
        start_time = datetime.now() 
        iterative_model.setParam('OutputFlag', 0)
        iterative_model.optimize()
        execution_time.append(datetime.now() - start_time)
        if iterative_model.status == gp.GRB.OPTIMAL:
            eps.append(threshold)
            of.append(iterative_model.objVal)
            changed_indices.append(indices)
            rows_changed.append(rows_zero)
            columns_changed.append(columns_zero)
            dv.append(np.array([var.x for var in iterative_model.getVars()]))
            variables = iterative_model.getVars()
            decisions = np.array([var.x for var in variables])
            abs_vio, obj_val = measuring_constraint_infeasibility(original_model, decisions)
            of_dec.append(obj_val)
            constraint_viol.append(abs_vio)
        else:
            eps.append(threshold)
            of.append(np.nan)  # Append NaN for objective function
            of_dec.append(np.nan)
            changed_indices.append(indices) # COMPROBATE THAT THIS MAKE SENSE
            rows_changed.append(rows_zero)
            columns_changed.append(columns_zero)
            constraint_viol.append(np.nan)
            dv.append(np.full(len(original_model.getVars()), np.nan))

        del iterative_model

        # Progress bar
        progress = iteration / total_iterations
        bar_length = 50  # Length of the progress bar
        progress_bar = '>' * int(bar_length * progress) + ' ' * (bar_length - int(bar_length * progress))
        sys.stdout.write(f"\r[{progress_bar}] Iteration {iteration}/{total_iterations}")
        sys.stdout.flush()

        iteration += 1  # Increment iteration counter
        threshold += step*threshold # Increment threshold

        
    results = {
        'epsilon': eps,
        'objective_function': of,
        'decision_variables': dv,
        'changed_indices': changed_indices,
        'rows_changed': rows_changed,
        'columns_changed': columns_changed,
        'constraint_violation': constraint_viol,
        'of_original_decision': of_dec,
        'non_zeros': total_non_zeros,
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
        standard_model = standard_form(model)
        print_model(original_primal)
        print_model(standard_model)
        original_primal.setParam('OutputFlag', 0)
        standard_model.setParam('OutputFlag', 0)
        original_primal.optimize()
        standard_model.optimize()
            # Check if the objective values are the same
        #if original_primal.Status == GRB.OPTIMAL and standard_model.Status == GRB.OPTIMAL:
        print(model_name[-1], original_primal.objVal,  standard_model.objVal)

def sensitivity_analysis_file(file,save_path):
    results = nested_dict()
    # for model_name in os.listdir(folder):
    #     if model_name.endswith('.xlsx') or model_name.endswith('modified.mps') or model_name == 'AJAX.mps':
    #         continue
    #     model_path = os.path.join(folder, model_name)
    model = gp.read(file)
    model_variables = add_new_restrictions_variables(model)
    model_variables.setParam('OutputFlag', 0)
    model_variables.optimize()
    print(model_variables.objVal)
    #model = gp.read(model_path)
    #original_primal, track_elements = canonical_form(model,minOption=True)
    original_primal = standard_form(model_variables)
    lb, ub = calculate_bounds(original_primal)
    
    # A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
    # save_json(A, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
    # Load the dual model
    model_to_use_dual = construct_dual_model(original_primal)

    presolve = load_class(model_variables)
    start_time = time.time()
    results_problem =  sensitivity_analysis_new(original_primal, presolve)
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
    results_problem_dual = sensitivity_analysis_new(model_to_use_dual, presolve_dual, dual = True)
    results[model_name]['dual'] = results_problem_dual
    results[model_name]['dual']['time_required'] = time.time() - start_time
    results[model_name]['dual']['mathematical_model'] = get_model_in_mathematical_format(
        model_to_use_dual)
    
    name = file.split('/')[-1]
    name = name.split('.')[0]
    model_save_path = os.path.join(save_path, f'epsilon_sparsification_{name}.json')
    #dict2json(results, model_save_path)
    save_pickle(results, f'epsilon_sparsification_{name}.pkl')
    return results

def sensitivity_analysis_folder(folder):
    results = nested_dict()
    for model_name in os.listdir(folder):
        if model_name.endswith('.xlsx') or model_name.endswith('modified.mps') or model_name == 'AJAX.mps':
            continue
        model_path = os.path.join(folder, model_name)
        model = gp.read(model_path)
        #original_primal, track_elements = canonical_form(model,minOption=True)
        original_primal = standard_form1(model)
        # A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
        # save_json(A, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
        # # Load the dual model
        model_to_use_dual = construct_dual_model(original_primal)

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
    objective_standard = 0

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
            model.setParam('OutputFlag', 0)
            model.optimize()
            print(model.objVal)
          
            if model.Status == gp.GRB.OPTIMAL:
                objective = model.objVal
                print(f'Objective value of the original model: {model.objVal}')
            standard_model = standard_form(model)
            standard_model.setParam('OutputFlag', 0)
            standard_model.optimize()
            #print(standard_model.objVal)
            A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(standard_model)
            save_json(A, b, c, lb, ub, of_sense, cons_senses,'model_matrices.json', co, variable_names)
            # Uncomment these if you want to calculate bounds later
     
            # lb, ub = calculate_bounds2(standard_model)
            bounds_model = build_model_from_json('model_matrices.json')
            bounds_model.setParam('OutputFlag', 0)
            bounds_model.optimize()

            dual_model = construct_dual_model(standard_model)
            dual_model.setParam('OutputFlag', 0)
            dual_model.optimize()
            if bounds_model.Status == gp.GRB.OPTIMAL:
                objective_bounds = bounds_model.objVal

            # # Check if the optimization status is optimal
            if standard_model.Status == gp.GRB.OPTIMAL:
                objective_standard = standard_model.objVal
                print(f'Objective value without bounds: {objective_standard}')
            else:
                objective_standard = 0  # Set to 0 if not optimal

            if dual_model.Status == gp.GRB.OPTIMAL:
                objective_dual = dual_model.objVal
                print(f'Objective value of the dual model: {objective_dual}')
          
            #del model, standard_model, bounds_model, dual_model

        except gp.GurobiError as e:
            print(f"Error reading or optimizing model {model_name}: {e}")
            objective_standard = 0  # If there's an error, set the objective to 0

        # Add a new row with model name, standard objective, and bounds objective

        new_row = [name, objective, objective_standard, objective_bounds, objective_dual]
        rows.append(new_row)
        
    # Create a DataFrame with the collected rows
    results = pd.DataFrame(rows, columns=['Model', 'Normal', 'Standard', 'Bounds', 'Dual'])
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
    # model_name = model_path_modified
    # print(model_name)
    # model = gp.read(model_name)

    # model.setParam('OutputFlag', 0)
    # model.optimize()
    # print(model.objVal)
    # # #print_model_in_mathematical_format(model)
    # # # Get the model in standard form
    # # #A, b, obj = parse_mps(model_name)
    # standard_model = standard_form(model)
    # # # #canonical_model, _ = canonical_form(model)
    # # # # canonical_model.setParam('OutputFlag', 0)
    # # #print_model_in_mathematical_format(standard_model)
    # standard_model.setParam('OutputFlag', 0)
    # standard_model.optimize()
    # print(standard_model.objVal)
    # # for var in standard_model.getVars():
    # #     print(f'{var.VarName}: {var.X}')
    # # # print(standard_model.objVal)
    # # print(standard_model.objVal)
    # # dual_model = construct_dual_model(standard_model)
    # # # # dual_model1 = construct_dual_model1(standard_model)
    # # # # print_model_in_mathematical_format(dual_model1)
    # # # # dual_model1.optimize()
    # # # # print(dual_model1.objVal)
    # # dual_model.setParam('OutputFlag', 0)
    # # dual_model.optimize()
    # # print(dual_model.objVal)


    # #print_model_in_mathematical_format(dual_model)
    # # for var in standard_model.getVars():
    # #     print(f'{var.VarName}: {var.X}')
    # # #print_model_in_mathematical_format(standard_model)
    # lb, ub = calculate_bounds(standard_model)
    # print(f'Lower bounds: {lb}')
    # print(f'Upper bounds: {ub}')

    # Count the number of infinite upper bounds
    #print(f'Number of infinite upper bounds: {len(t[0])}')
    # # # # # # See if lb <= ub
    # # # # # print(all(lb_new <= ub_new))
    # path = os.path.join(os.getcwd(), 'model_matrices.json')
    # model1 = build_model_from_json(path)
    # # # print_model_in_mathematical_format(model)
    # model1.setParam('OutputFlag', 0)
    # model1.optimize()
    # print(model1.objVal)
    # for var in model.getVars():
    #     print(f'{var.VarName}: {var.X}')
    # Compute the sensitivity analysis

    #main_bounds(GAMS_path_modified)

    """
    ----------------SENSITIVITY ANALYSIS FOR A SINGLE MODEL----------------
    """
    # model_names =['AIRSP', 'PRODMIX', 'SPARTA', 'AIRCRAFT', 'SRKANDW', 'UIMP', 'PORT', 'GUSSEX1','GUSSGRID','SENSTRAN','TRNSPORT']
    # model_names_2 = ['DANWOLFE.mps', 'DINAM.mps', 'EGYPT.mps', 'INDUS89.mps','FERTS.mps', 'ISWNM.mps', 'MSM.mps', 'PHOSDIS.mps', 'PRODSP.mps', 'SARF.mps', 'TABORA.mps', 'TURKPOW.mps']


    # all_results = nested_dict()
    sparsification_results = os.path.join(project_root, 'results/epsilon_sparsification_variables1')
    results_folder = os.path.join(project_root, 'results/epsilon_sparsification')
    zeroepsilon_row_results = os.path.join(project_root, 'results/zeroepsilon_rows_norm')
    # if not os.path.exists(zeroepsilon_row_results):
    #     os.makedirs(zeroepsilon_row_results)
    # # # Create a list with the models to analyze
    # # model_names = []
    # # for model_name in os.listdir(GAMS_path_modified):
    # #     if model_name.endswith('.xlsx'):
    # #         continue
    # #     name = model_name.split('.')[0]
    # #     model_names.append(name)

    # # # Get a set of already analyzed model names from results_folder
    # # analyzed_models = set()
    # # for result_file in os.listdir(results_folder):
    # #     # Assuming the result files are named in a way that includes the model name
    # #     name = result_file.split('.')[0]  # Remove file extension
    # #     name = name.split('_')[-1]        # Extract model name part if your files are named like 'result_modelname.txt'
    # #     analyzed_models.add(name)

    # # # Filter out the models that have already been analyzed
    # # models_to_process = [model for model in os.listdir(GAMS_path_modified) if model not in model_names_2]
    # # #models_to_process = model_names_2
    # # # print(len(models_to_process))
    # # # print(len(model_names_2))
    # # # for model_name in models_to_process:
    # # #     if model_name == 'INDUS89.mps':
    # # #         continue
    # # #     model_path = os.path.join(GAMS_path_modified, f'{model_name}')
    # # #     results = sensitivity_analysis_file(model_path,save_path = sparsification_results)
    # # sensitivity_analysis_file(model_path_modified,save_path = zeroepsilon_row_results)
    # # # # if not os.path.exists(sparsification_results):
    # # #     os.makedirs(sparsification_results)
    # for name in os.listdir(GAMS_path_modified):
    #     # if name == 'INDUS89.mps':
    #     #     continue
    #     model_path = os.path.join(GAMS_path_modified, f'{name}')
    #     results = sensitivity_analysis_file(model_path,save_path = sparsification_results)
    #     #all_results.update(results)
    sensitivity_analysis_file(model_path_modified,save_path = sparsification_results)
    # #dict2json(all_results, 'epsilon_sparsification.json')
 
    # # # model_name = os.path.join(GAMS_path_modified, 'DINAM.mps')
    # # # model = gp.read(model_name)
    # # #original_primal, track_elements = canonical_form(model)
    # # #standar_model = standard_form2(model)
    # # # # presolve = load_class(standar_model)
    # # # results_problem =  sensitivity_analysis(standar_model, presolve)
    # # # dict2json(results_problem, 'epsilon_results.json')
    # # # print(results_problem)
    # # #sensitivity_analysis_folder(GAMS_path_modified)

