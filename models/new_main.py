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
model_path = os.path.join(project_root, 'data/GAMS_library', 'AIRSP.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')

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



def sensitivity_analysis(model, presolve, min_threshold = 0.0015, max_threshold = 0.3, step = 0.2): 
    
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

    (A, b, c, lb, ub, of_sense, cons_senses, co, variable_names,
                change_dictionary, operation_table)   = presolve.orchestrator_presolve_operations(model = original_model)
    model = build_normalized_model(A, b, c, lb, ub, of_sense, cons_senses, variable_names)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print('Optimal objective value (normalized model):', model.ObjVal)
        solution = model.getAttr('X', model.getVars())
    # for var_name, value in zip(variables_names, solution):
    #     print(f'{var_name}: {value}')
    else:
        print('Optimization was not successful.')

    A_initial = A.copy()
    print(of_sense)
    threshold = min_threshold
    while threshold <= max_threshold: 
        start_time = datetime.now()
        (A, b, c, lb, ub, of_sense, cons_senses, co, variable_names,
                change_dictionary, operation_table)   = presolve.orchestrator_presolve_operations(model = original_model)
        A_changed = A.copy()
        #print(np.all(of_sense == cons_senses_initial))
                    
        indices = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1]) if
                       A_initial[i, j] != 0 and A_changed[i, j] == 0]
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
            #changed_indices.append(indices)
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



if __name__ == '__main__':
    results = nested_dict()
    for model_name in os.listdir(GAMS_path):
        if model_name.endswith('.xlsx'):
            continue
        model = gp.read(model_path)
        original_primal, track_elements = canonical_form(model)

        presolve = load_class(original_primal)
        results_problem =  sensitivity_analysis(original_primal, presolve)
        results[model_name] = results_problem

    dict2json(results, 'epsilon_results.json')