import json 
import os 
import gurobipy as gp
import numpy as np 

from utils_models.utils_functions import * 
from utils_models.standard_model import *



project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_folder = os.path.join(project_root, 'results/prueba')
file = os.path.join(results_folder, 'epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')
model_file_new = os.path.join(project_root, 'data/real_data_new/openTEPES_EAPP_2030_sc01_st1.mps')
model_file = os.path.join(project_root, 'data/real_data/openTEPES_9n_2030_sc01_st1.mps')

GAMS_path = os.path.join(project_root, 'data/GAMS_library')    
real_data_path = os.path.join(project_root, 'data/real_data_new')
import gurobipy as gp

def set_real_objective(model):
    """
    This function assumes the model has an 'objective variable' obj_var
    (i.e., the model's objective is something like Minimize obj_var),
    and that there's a single constraint which links obj_var to the real
    linear expression of other variables. It identifies that constraint,
    reconstructs the real objective, and replaces the model's objective
    with the unwrapped expression. Then it removes both the old linking
    constraint and the old auxiliary objective variable from the model.

    Returns: The same model object, but with the "real" objective set,
             the old linking constraint removed, and the old obj_var removed.
    """

    # --- 1) Get the current objective ---
    objective_expr = model.getObjective()
    if objective_expr.size() != 1:
        print("The current objective is not a single variable term.")
        print("No changes made to the model.")
        return model  # Early return

    # The objective is (obj_coef * obj_var). Often obj_coef = 1.0
    obj_var = objective_expr.getVar(0)
    obj_coef = objective_expr.getCoeff(0)

    print(f"Detected auxiliary objective variable: {obj_var.VarName}")
    print(f"Objective = {obj_coef} * {obj_var.VarName}")

    # --- 2) Search for the linking constraint where obj_var appears ---
    found_linking_constraint = False
    linking_constr_to_remove = None

    for constr in model.getConstrs():
        row_expr = model.getRow(constr)  # LHS of the constraint
        sense = constr.Sense            # '=', '<', or '>'
        rhs = constr.RHS                # numeric right-hand side

        obj_var_coeff = 0.0
        other_terms_expr = gp.LinExpr()

        # Scan all terms in this constraint
        for i in range(row_expr.size()):
            v = row_expr.getVar(i)
            c = row_expr.getCoeff(i)

            # Compare object references to detect obj_var
            if v is obj_var:
                obj_var_coeff = c
            else:
                other_terms_expr.addTerms(c, v)

        # We assume a single constraint has the form:
        #   (obj_var_coeff * obj_var) + sum(...) = rhs
        # so that obj_var = (rhs - sum(...)) / obj_var_coeff
        if obj_var_coeff != 0 and sense == '=':
            found_linking_constraint = True
            linking_constr_to_remove = constr

            print(f"Found linking constraint: {constr.ConstrName}")
            #print(f"LHS: {row_expr} {sense} {rhs}")
            
            # --- 3) Reconstruct the "real" objective expression ---
            real_obj_expr = gp.LinExpr()
            
            # Start with RHS / obj_var_coeff
            real_obj_expr.addConstant(rhs / obj_var_coeff)

            # Then subtract (1/obj_var_coeff) * other_terms_expr
            # Because:
            #    obj_var_coeff * obj_var + other_terms_expr = rhs
            # -> obj_var = (rhs - other_terms_expr) / obj_var_coeff
            real_obj_expr.add(other_terms_expr, -1.0 / obj_var_coeff)

            #print(f"Real objective expression: {obj_var.VarName} = {real_obj_expr}")

            # # --- 4) Replace the model's objective with this expression ---
            # if model.ModelSense == gp.GRB.MINIMIZE:
            #     model.setObjective(real_obj_expr, gp.GRB.MINIMIZE)
            # else:
            #     model.setObjective(real_obj_expr, gp.GRB.MAXIMIZE)

            print("Model's objective has been replaced with the real objective.")
            break  # Stop after the first valid linking constraint
    
    # Put the new objective in the model
    model.setObjective(real_obj_expr, gp.GRB.MINIMIZE)
    model.update()
    # --- 5) Remove the linking constraint and the auxiliary objective variable ---
    if found_linking_constraint:
         model.remove(linking_constr_to_remove)
         model.remove(obj_var)
    #     print("Linking constraint and auxiliary objective variable have been removed.")
    # else:
    #     print("No valid linking constraint found. No changes made to the model.")
    model.update()
    obj = model.getObjective()
    num_vars = obj.size()
    print(num_vars)
    return model

def euclidean_norm(A, c, column):
    """
    Compute the euclidean norm of the columns of a matrix A
    """
    

if __name__ == '__main__':
    GAMS_path = os.path.join(project_root, 'data/real_data')
    problem = 'openTEPES_9n_2030_sc01_st1.mps'
    model_file = os.path.join(GAMS_path, problem)
    model = gp.read(model_file)
    model_new = set_real_objective(model)
    model_new.setParam('OutputFlag', 0)
    # model_new.optimize()
    # print(model_new.ObjVal)
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model_new)
    # Get the values of c where it is non-zero
    c = np.array(c)
    b = np.array(b)
    non_zero_c = np.where((c != 1) & (c != 0))[0]   
    print(len(non_zero_c))
    euclidean_norm(A,b,17472)
    # Get the objective function
    # obj = model_new.getObjective()
    # print(obj)
