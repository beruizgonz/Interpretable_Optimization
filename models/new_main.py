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
#opts.eliminate_zero_rows_epsilon = True


#The path to the data
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'INDUS.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'PHOSDIS.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_EAPP_2030_sc01_st1.mps')

def load_class(model):
    presolve = PresolvepsilonOperations(model, eliminate_zero_rows_epsilon = True, opts = opts)
    return presolve

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


def sensitivity_analysis(model, presolve, dual=False, min_threshold=0.015, max_threshold=0.3, step=0.2):
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
    abs_vio, obj_val = measuring_constraint_infeasibility(model, dv[0])
    of = [obj_val]
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
        (A_new, b, c, lb, ub, of_sense, cons_senses, co, variable_names,
        change_dictionary, operation_table) = presolve.orchestrator_presolve_operations(
            model=original_model, epsilon=threshold
        )
        non_zeros = A_initial.nonzero()
        total_non_zeros = len(non_zeros[0])
        print('Non zero: ', total_non_zeros)
        A_changed = csr_matrix(A_new)  # Use sparse matrix
        indices = [(int(i), int(j)) for i, j in zip(*non_zeros) if A_changed[i, j] == 0]
        rows_zero = np.where(~A_changed.toarray().any(axis=1))[0]
        columns_zero = np.where(~A_changed.toarray().any(axis=0))[0]
        #print(np.all(A_initial == A_new))
        if len(indices) > 0:
            print('Changed: ', len(indices))
        
        save_json(A_changed, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
        # print(lb, ub)
        # print(cons_senses)
        #iterative_model = (construct_dual_model_from_json if dual else construct_model_from_json)(opts.save_path)
        iterative_model = build_model_from_json(opts.save_path)
        start_time = datetime.now()
        iterative_model.setParam('OutputFlag', 0)
        iterative_model.optimize()
        execution_time.append(datetime.now() - start_time)

        if iterative_model.status == gp.GRB.OPTIMAL:
            print('Optimal solution found: %s' % (iterative_model.objVal))
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
    original_primal = standard_form(model)
    original_primal.setParam('OutputFlag', 0)
    original_primal.optimize()
   
    # Sove orignal model
    # A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
    # save_json(A, b, c, lb, ub, of_sense, cons_senses, opts.save_path, co, variable_names)
    # Load the dual model
    model_to_use_dual = construct_dual_model(original_primal)

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
    #results_problem_dual = sensitivity_analysis_new(model_to_use_dual, presolve_dual, dual = True)
    # solve the dual model
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


if __name__ == '__main__':
    """
    ----------------SENSITIVITY ANALYSIS FOR A SINGLE MODEL----------------
    """
    # model_names =['AIRSP', 'PRODMIX', 'SPARTA', 'AIRCRAFT', 'SRKANDW', 'UIMP', 'PORT', 'GUSSEX1','GUSSGRID','SENSTRAN','TRNSPORT']
    # model_names_2 = ['DANWOLFE.mps', 'DINAM.mps', 'EGYPT.mps', 'INDUS89.mps','FERTS.mps', 'ISWNM.mps', 'MSM.mps', 'PHOSDIS.mps', 'PRODSP.mps', 'SARF.mps', 'TABORA.mps', 'TURKPOW.mps']


    # all_results = nested_dict()
    sparsification_results = os.path.join(project_root, 'results/epsilon_sparsification')
    results_folder = os.path.join(project_root, 'results/epsilon_sparsification')
    zeroepsilon_row_results = os.path.join(project_root, 'results/zeroepsilon_rows_norm_variables')
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
    for name in os.listdir(GAMS_path_modified):
        if name == 'INDUS89.mps':
            continue
        model_path = os.path.join(GAMS_path_modified, f'{name}')
        results = sensitivity_analysis_file(model_path,save_path = sparsification_results)
        #all_results.update(results)
    #sensitivity_analysis_file(real_model_path,save_path = zeroepsilon_row_results)
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

