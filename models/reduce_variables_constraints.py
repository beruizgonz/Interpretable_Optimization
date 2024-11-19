import gurobipy as gp
import datetime
import numpy as np
import math
from scipy.sparse import csr_matrix
import sys
import os 

from utils_models.utils_functions import * 
from utils_models.standard_model import *


# ACCESS TO THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'INDUS.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'TURKPOW.mps')


def reduce_constraints(problem, epsilon = 0.00014):
    model = gp.read(problem)
    standard_model = standard_form(model)
    # Construct the dual model
    dual_standard_model = construct_dual_model(standard_model)
    # Solve the dual model
    dual_standard_model.setParam('OutputFlag', 0)
    dual_standard_model.optimize()
    # Get the dual variables    
    dual_variables = np.array([var.x for var in dual_standard_model.getVars()])
    # Get the maximum dual variable value
    max_dual = np.max(dual_variables)
    print('Max dual:', max_dual*epsilon)
    print('Max dual:', max_dual)    
    # Get the indices of the dual variables that are close to the maximum dual variable value
    indices = np.where(dual_variables >= max_dual*epsilon)[0]
    # Get the variables that less than 0. 
    indices_neg = np.where(dual_variables < 0)[0]
    print('Indices neg:', len(indices_neg))
    min_dual = np.min(dual_variables)
    print('Min dual:', min_dual)
    print('Number of variables:', len(standard_model.getVars()))
    # Get the constraints that are close to the maximum dual variable value
    print('Number of dual variables:', len(dual_variables))
    print('Number of constraints to mantain:', len(indices))
    # Number of constraints in the dual model
    print('Number of constraints in the dual model:', len(dual_standard_model.getConstrs()))
    # The indices of the constraints to mantain in the standard model
    # Retrieve all constraints once to avoid multiple calls
    reduced_model = standard_model.copy()
    all_constraints = standard_model.getConstrs().copy()
    indices_set = set(indices)

    all_indices_set = set(range(len(all_constraints)))

# Indices of constraints to remove
    indices_set_to_remove = all_indices_set - indices_set

    all_constraint_names = [constr.ConstrName for constr in all_constraints]
    constraints_to_remove = [
        all_constraint_names[idx]
        for idx  in indices_set_to_remove
        if 0 <= idx < len(all_constraint_names)
    ]
    
    for name in constraints_to_remove:
        try:
            constraint = reduced_model.getConstrByName(name)
            if constraint:
                reduced_model.remove(constraint)
            else:
                print(f"Constraint '{name}' not found in the reduced_model.")
        except gp.GurobiError as e:
            print(f"Error removing constraint '{name}': {e}")
            
    return dual_standard_model.objVal, reduced_model    


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


if __name__ == '__main__':
    objective, reduced_model = reduce_constraints(model_path_modified)
    print(objective)
    reduced_model.setParam('OutputFlag', 0) 
    reduced_model.optimize()
    constraints = reduced_model.getConstrs()
    print('Number of constraints:', len(constraints))
    if reduced_model.status == gp.GRB.OPTIMAL:
        print('Optimal solution found: %s' % (reduced_model.objVal))
