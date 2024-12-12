import json
import time
import os
import gurobipy as gp
import logging
import sys
from datetime import datetime
import math
from scipy.sparse import coo_matrix, csr_matrix

from opts import parse_opts
from utils_models.presolvepsilon_class import PresolvepsilonOperations
from utils_models.utils_functions import *
from utils_models.standard_model import standard_form_e1, construct_dual_model_sparse
 
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

# Load the data information
opts = parse_opts()

# PATH TO THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'INDUS.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')
bounds_path = os.path.join(project_root, 'data/bounds_trace')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'DANWOLFE.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_9n_2030_sc01_st1.mps')

# PATH TO SAVE THE RESULTS
results_folder = os.path.join(project_root, 'results_new')
sparsification_results = os.path.join(results_folder, 'sparsification_optimal_solution')
zeroepsilon_row_results = os.path.join(results_folder, 'epsilon_rows_norm')
zeroepsilon_col_results = os.path.join(results_folder, 'epsilon_cols_norm')
prueba = os.path.join(results_folder, 'prueba')


def load_class(model, opts):
    presolve = PresolvepsilonOperations(model, opts = opts)
    return presolve

def sensitivity_analysis(model, presolve, dual=False, min_threshold=0.015, max_threshold=0.3, step=0.2):
    """
    Perform sensitivity analysis on a Gurobi model by iteratively modifying constraints and variables.

    Parameters:
    - model: Gurobi model object.
    - presolve: Presolve object to apply orchestrated presolve operations.
    - dual: Boolean indicating if the dual model should be constructed.
    - min_threshold: Minimum epsilon threshold for presolve operations.
    - max_threshold: Maximum epsilon threshold for presolve operations.
    - step: Multiplicative step for threshold increment.

    Returns:
    - results: Dictionary containing sensitivity analysis results.
    """
    model.setParam('OutputFlag', 0)
    model.optimize()

    # Initialize variables
    eps = [0]
    execution_time = [0]
    dv = [np.array([var.x for var in model.getVars()])]
    obj_val = model.objVal
    print(f'Optimal solution found: {obj_val}')

    abs_vio, _ = measuring_constraint_infeasibility1(model, dv[0])

    of = [obj_val]
    constraint_viol = [abs_vio]
    of_dec = [obj_val]
    original_model = model.copy()

    # Extract model matrices and metadata
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model=original_model)
    A_initial = csr_matrix(A)
    total_non_zeros = A_initial.nnz

    # Calculate total iterations for progress bar
    total_iterations = math.ceil(math.log(max_threshold / min_threshold) / math.log(1 + step))

    results = {
        'epsilon': eps,
        'objective_function': of,
        'decision_variables': dv,
        'changed_indices': [None],
        'rows_changed': [None],
        'columns_changed': [None],
        'constraint_violation': constraint_viol,
        'of_original_decision': of_dec,
        'non_zeros': total_non_zeros,
        'execution_time': execution_time,
    }

    threshold = min_threshold
    iteration = 1

    while threshold <= max_threshold:
        print(f"Threshold: {threshold}")

        # Apply presolve operations
        A_new, b_new, _, _, _, _, _, _, _, _, _ = presolve.orchestrator_presolve_operations(
            model=original_model, epsilon=threshold
        )
        
        A_changed = csr_matrix(A_new)
        indices = [(i, j) for i, j in zip(*A_initial.nonzero()) if A_changed[i, j] == 0]
        zero_columns = np.where(A_changed.getnnz(axis=0) == 0)[0]
        zero_rows = np.where(A_changed.getnnz(axis=1) == 0)[0]

        print(f'Changed: {len(indices)}, Zero columns: {len(zero_columns)}, Zero rows: {len(zero_rows)}')

        # Build and optimize new model
        iterative_model = build_model(A_new, b_new, c, co, lb, ub, of_sense, cons_senses, variable_names)
        start_time = datetime.now()

        iterative_model.setParam('OutputFlag', 0)
        iterative_model.optimize()

        execution_time.append(datetime.now() - start_time)

        # Process optimization results
        if iterative_model.status == GRB.OPTIMAL:
            results['epsilon'].append(threshold)
            results['objective_function'].append(iterative_model.objVal)
            results['decision_variables'].append(np.array([var.x for var in iterative_model.getVars()]))
            results['changed_indices'].append(indices)
            results['rows_changed'].append(zero_rows)
            results['columns_changed'].append(zero_columns)
            abs_vio, _ = measuring_constraint_infeasibility1(original_model, results['decision_variables'][-1])
            results['constraint_violation'].append(abs_vio)
            results['of_original_decision'].append(iterative_model.objVal)
        else:
            print("Model is infeasible or unbounded.")
            results['epsilon'].append(threshold)
            results['objective_function'].append(np.nan)
            results['decision_variables'].append(np.full(len(original_model.getVars()), np.nan))
            results['changed_indices'].append(indices)
            results['rows_changed'].append(zero_rows)
            results['columns_changed'].append(zero_columns)
            results['constraint_violation'].append(np.nan)
            results['of_original_decision'].append(np.nan)

        del iterative_model

        # Update progress bar
        progress = iteration / total_iterations
        sys.stdout.write(f"\r[{'>' * int(50 * progress):<50}] Iteration {iteration}/{total_iterations}")
        sys.stdout.flush()

        iteration += 1
        threshold += step * threshold

    return results

def sensitivity_analysis1(model, presolve, dual=False, min_threshold=0.015, max_threshold=0.3, step=0.2):
    model.setParam('OutputFlag', 0)
    model.optimize()
    eps = [0]
    execution_time = [0]
    print('Optimal solution found: %s' % (model.objVal))
    obj_val = model.objVal
    dv = [np.array([var.x for var in model.getVars()])]
    changed_indices = [None]
    rows_changed = [None]
    columns_changed = [None]
    abs_vio, obj_val = measuring_constraint_infeasibility1(model, dv[0])
    of = [model.objVal]
    constraint_viol = [abs_vio]
    of_dec = [model.objVal]
    iteration = 1
    original_model = model.copy()
    (A, b, c, co, lb, ub, of_sense, cons_senses, variable_names) = get_model_matrices(model=original_model)
    A_initial = csr_matrix(A)  # Use sparse matrix
    non_zeros = A_initial.nonzero()
    total_non_zeros = len(non_zeros[0])
    threshold = min_threshold
    
    # Calculate total iterations for progress bar
    total_iterations = math.ceil(math.log(max_threshold / min_threshold) / math.log(1 + step))
    while threshold <= max_threshold:
        print(f"Threshold: {threshold}")
        (A_new, b_new, c, lb, ub, of_sense, cons_senses, co, variable_names,
        change_dictionary, operation_table) = presolve.orchestrator_presolve_operations(
            model=original_model, epsilon=threshold
        )
        are_equal = (A_initial != A_new).nnz == 0
        print('Are equal: ', are_equal)
        A_new = csr_matrix(A_new)  # Use sparse matrix
        non_zeros = A_initial.nonzero()
        total_non_zeros = len(non_zeros[0])
        print('Non zero: ', total_non_zeros)
        A_changed = csr_matrix(A_new)  # Use sparse matrix
        indices = [(int(i), int(j)) for i, j in zip(*non_zeros) if A_changed[i, j] == 0]
        zero_columns = np.where(A_changed.getnnz(axis=0) == 0)[0]
        zero_rows = np.where(A_changed.getnnz(axis=1) == 0)[0]
        #print(np.all(A_initial == A_new))
        if len(indices) > 0:
            print('Changed: ', len(indices))
        print('Zero columns: ', len(zero_columns))
        print('Zero rows: ', len(zero_rows))
        
        #save_json(A_changed, b_new, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
        # print(lb, ub)
        # print(cons_senses)
        #iterative_model = (construct_dual_model_from_json if dual else construct_model_from_json)(opts.save_path)
        iterative_model = build_model(A_new, b_new, c, co, lb, ub, of_sense, cons_senses, variable_names)
        start_time = datetime.now()
        iterative_model.setParam('OutputFlag', 0)
        iterative_model.optimize()
        execution_time.append(datetime.now() - start_time)
        if iterative_model.status == GRB.INFEASIBLE:
            print("The model is infeasible.")
        elif iterative_model.status == GRB.UNBOUNDED:
            print("The model is unbounded.")

        if iterative_model.status == gp.GRB.OPTIMAL:
            print('Optimal solution found: %s' % (iterative_model.objVal))
            eps.append(threshold)
            of.append(iterative_model.objVal)
            changed_indices.append(indices)
            rows_changed.append(zero_rows)
            columns_changed.append(zero_columns)
            dv.append(np.array([var.x for var in iterative_model.getVars()]))
            variables = iterative_model.getVars()
            decisions = np.array([var.x for var in variables])
            abs_vio, obj_val = measuring_constraint_infeasibility1(original_model, decisions)
            of_dec.append(iterative_model.objVal)
            constraint_viol.append(abs_vio)
        else:
            eps.append(threshold)
            of.append(np.nan)
            of_dec.append(np.nan)
            changed_indices.append(indices)
            rows_changed.append(zero_rows)
            columns_changed.append(zero_columns)
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

def sensitivity_analysis_file(file, save_path, opts):
    """
    Perform sensitivity analysis for a given model file and save results.

    Parameters:
    - file: Path to the model file.
    - save_path: Directory to save the results.
    - opts: Options object with configurations.

    Returns:
    - results: Nested dictionary containing sensitivity analysis results.
    """
    results = nested_dict()
    model = gp.read(file)
    model_name = os.path.splitext(os.path.basename(file))[0]

    # Convert the model to standard form
    original_primal = standard_form_e1(model)

    # Handle bounds
    if opts.read_bounds:
        A_sparse, b_sparse, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
        json_bounds = os.path.join(project_root, 'data/bounds_trace', f'bound_trace_{model_name}.json')
        with open(json_bounds, 'r') as f:
            data = json.load(f)
        last_iteration_index = len(data['iteration']) - 1
        lb, ub = data['lb'][last_iteration_index], data['ub'][last_iteration_index]
    else:
        A_sparse, b_sparse, c, co, lb, ub, of_sense, cons_senses, variable_names = calculate_bounds_candidates_sparse(original_primal, None, model_name)

    # Build model with bounds
    model_bounds = build_model(A_sparse, b_sparse, c, co, lb, ub, of_sense, cons_senses, variable_names)

    # Perform primal sensitivity analysis
    presolve = load_class(model_bounds, opts=opts)
    start_time = time.time()
    results_problem = sensitivity_analysis1(model_bounds, presolve, opts)
    results[model_name]['primal'] = results_problem
    results[model_name]['primal']['time_required'] = time.time() - start_time

    # Perform dual sensitivity analysis
    log.info(f"{str(datetime.now())}: Sensitivity Analysis with dual model...")
    model_to_use_dual = construct_dual_model_sparse(original_primal)
    start_time = time.time()

    presolve_dual = load_class(model_to_use_dual, opts=opts)
    model_to_use_dual.setParam('OutputFlag', 0)
    model_to_use_dual.optimize()

    if model_to_use_dual.status == gp.GRB.OPTIMAL:
        dv_dual = np.array([var.x for var in model_to_use_dual.getVars()])
        results_problem_dual = {
            'objective_function': model_to_use_dual.objVal,
            'decision_variables': dv_dual
        }
        results[model_name]['dual'] = results_problem_dual
        results[model_name]['dual']['time_required'] = time.time() - start_time

    # Determine operation type and save results
    operation = 'rows' if opts.operate_epsilon_rows else ('cols' if opts.operate_epsilon_cols else 'both')
    model_save_path = os.path.join(save_path, f'epsilon_{operation}_{model_name}.json')
    dict2json(results, model_save_path)

    return results

if __name__ == '__main__':
    """
    ----------------SENSITIVITY ANALYSIS FOR A SINGLE MODEL----------------
    """

    # # # Filter out the models that have already been analyzed
    # models_to_process = [model for model in os.listdir(GAMS_path_modified) if model not in model_names_2]
    # #models_to_process = model_names_2
    # # print(len(models_to_process))
    # # print(len(model_names_2))
    # # for model_name in models_to_process:
    # #     if model_name == 'INDUS89.mps':
    # #         continue
    # #     model_path = os.path.join(GAMS_path_modified, f'{model_name}')
    # #     results = sensitivity_analysis_file(model_path,save_path = sparsification_results)
    # sensitivity_analysis_file(model_path_modified,save_path = zeroepsilon_row_results)
    # # # if not os.path.exists(sparsification_results):
    # #     os.makedirs(sparsification_results)
    for name in os.listdir(GAMS_path_modified):
        if name == 'INDUS89.mps':
            continue
        model_path = os.path.join(GAMS_path_modified, f'{name}')
        opts.operate_epsilon_rows = True
        results = sensitivity_analysis_file(model_path,save_path = zeroepsilon_row_results, opts = opts)
        print('CHANGE TO COLUMNS')
        opts.operate_epsilon_cols = True
        opts.operate_epsilon_rows = False
        results = sensitivity_analysis_file(model_path,save_path = zeroepsilon_col_results, opts = opts)

    # Real problem path    
    # model = gp.read(real_model_path)
    # opts.operate_epsilon_rows = True
    # opts.operate_epsilon_cols = False
    # sensitivity_analysis_file(real_model_path, save_path = prueba, opts = opts)
    # print('CHANGE TO COLUMNS')
    # opts.operate_epsilon_cols = True
    # opts.operate_epsilon_rows = False   
    # sensitivity_analysis_file(real_model_path, save_path = prueba, opts = opts)
    
