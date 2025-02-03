import os
import gurobipy as gp
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# PATH TO THE DATA
parent_path = os.path.dirname(os.getcwd())
root_interpretable_optimization = os.path.dirname(parent_path)


# IMPORT FILES THAT ARE IN THE ROOT DIRECTORY
sys.path.append(parent_path)
from  models.utils_models.utils_functions import *
from models.utils_models.standard_model import *
from plot_matrix import *

# PATH TO THE DATA
real_data_path = os.path.join(parent_path, 'data/real_data')
open_tepes_9n = os.path.join(real_data_path, 'openTEPES_EAPP_2030_sc01_st1.mps')

# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/global/sparse/real_problems')
results_sparsification_open_tepes_9n = os.path.join(results_folder, 'sparsification/epsilon_sparsification_openTEPES_EAPP_2030_sc01_st1.json')
results_cols_open_tepes_9n = os.path.join(results_folder, 'epsilon_cols_abs/epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')
results_rows_open_tepes_9n = os.path.join(results_folder, 'epsilon_rows_abs/epsilon_rows_openTEPES_EAPP_2030_sc01_st1.json')

def read_json(file):
    """
    Read the json file
    """
    with open(file) as f:
        data = json.load(f)

    primal_model = data['openTEPES_EAPP_2030_sc01_st1']['primal']
    rows_changed = primal_model['rows_changed']
    columns_changed = primal_model['columns_changed']
    indices_changed = primal_model['changed_indices']
    epsilons = primal_model['epsilon']
    obj_value = primal_model['objective_function']
    return rows_changed, columns_changed, indices_changed, epsilons, obj_value

def reconstruct_model(model, epsilon_number, file):
    """
    Reconstruct the model
    """
    rows_changed, columns_changed, indices_changed, epsilons, obj_value = read_json(file)
    model = gp.read(model)
    standar_model = standard_form_e1(model)
    constraint_names = [constr.ConstrName for constr in standar_model.getConstrs()]
    A_sparse, b_sparse, c, co, lb, ub, of_sense, cons_senses, variable_names = calculate_bounds_candidates_sparse(standar_model, None, None)
    indices_changes_epsilon = indices_changed[epsilon_number]
    rows_changed_epsilon = rows_changed[epsilon_number]
    columns_changed_epsilon = columns_changed[epsilon_number]

    # Modify A_sparse and b_sparse
    A_sparse = A_sparse.tolil()  # Convert to LIL format for modification
    for index in indices_changes_epsilon:
        A_sparse[index[0], index[1]] = 0
    A_sparse = A_sparse.tocsr()  # Convert back to CSR format

    # # Debugging after modification
    # print("Sparse matrix A after modification:\n", A_sparse)
    # print("Sparse vector b after modification:\n", b_sparse)

    # Reconstruct the model
    print('Reconstructing the model')
    print("Objective value for epsilon:", obj_value[epsilon_number])

    reconstructed_model = build_model(A_sparse, b_sparse, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names)
    return reconstructed_model

def infeasiblity_constraints(model):
    """
    Check if the model is infeasible
    """
    model.setParam('OutputFlag', 0)
    model.optimize()

    if model.Status == gp.GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")

    # Compute IIS
    model.computeIIS()
    
    # Write IIS report to a file
    model.write("infeasibility_report.ilp")

    # Inspect IIS constraints and variables
    print("\nIIS Constraints:")
    iss_constraints= []
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"Constraint {c.ConstrName} is in the IIS.")
            iss_constraints.append(c.ConstrName)


    print("\nIIS Variable Bounds:")
    # for v in model.getVars():
    #     if v.IISLB:
    #         print(f"Variable {v.VarName} has its lower bound in the IIS.")
    #     if v.IISUB:
    #         print(f"Variable {v.VarName} has its upper bound in the IIS.")
    return iss_constraints

def remove_infeasible_constraints(model, iss_constraints):
    """
    Remove the infeasible constraints from the model.
    """
    print("Removing infeasible constraints...")
    for constr in iss_constraints:
        constr_to_remove = model.getConstrByName(constr)
        if constr_to_remove:
            model.remove(constr_to_remove)
            print(f"Removed constraint: {constr}")
        else:
            print(f"Constraint {constr} not found in the model.")

    # Update the model and re-optimize
    model.update()
    print("Model updated. Re-optimizing...")
    model.optimize()  
    if model.Status == gp.GRB.INFEASIBLE:
        iss_constraints = infeasiblity_constraints(model)
        feasible_model = remove_infeasible_constraints(model, iss_constraints)
    elif model.Status == gp.GRB.OPTIMAL:
        print("Model is feasible.")
        print("Objective value:", model.objVal)
    return model


if __name__ == '__main__':
    reconstructed_model = reconstruct_model(open_tepes_9n, 7,  results_sparsification_open_tepes_9n)
    iss_constrs = infeasiblity_constraints(reconstructed_model)
    feasible_model = remove_infeasible_constraints(reconstructed_model, iss_constrs)