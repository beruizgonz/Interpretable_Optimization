import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils_models.utils_functions import *

def standard_form1(model):
    # Step 1: Ensure the model is in minimization form
    if model.ModelSense != 1:
        model.setObjective(-1 * model.getObjective(), GRB.MINIMIZE)
        model.ModelSense = 1  # Set the model sense to minimization

    # Step 2: Ensure all variables are non-negative
    for var in model.getVars():
        # Create a new variable with lb=0 and ub=+infinity, same type as 'var'
        var_new = model.addVar(lb=0, vtype=var.VType, name=f"{var.VarName}_new")
        model.update()

        # Add constraints to enforce original bounds
        if var.LB > -GRB.INFINITY:
            model.addConstr(var_new >= var.LB, name=f"{var.VarName}_LB")
        if var.UB < GRB.INFINITY:
            model.addConstr(var_new <= var.UB, name=f"{var.VarName}_UB")

        # Replace 'var' in constraints
        for constr in model.getConstrs():
            coeff = model.getCoeff(constr, var)
            if coeff != 0:
                model.chgCoeff(constr, var_new, coeff)
                model.chgCoeff(constr, var, 0)

        # Replace 'var' in the objective function
        obj_coeff = var.Obj
        if obj_coeff != 0:
            model.setObjective(model.getObjective() + obj_coeff * var_new, GRB.MINIMIZE)

        # Remove the original variable from the model
        model.remove(var)

    model.update()

    # Step 3: Transform all constraints into equalities by introducing slack/surplus variables
    for constr in model.getConstrs():
        sense = constr.Sense
        if sense != GRB.EQUAL:
            # Add slack or surplus variable based on constraint type
            slack_var = model.addVar(lb=0, name=f"slack_{constr.ConstrName}")
            model.update()

            # Add slack or surplus depending on the sense of the constraint
            if sense == GRB.LESS_EQUAL:
                model.chgCoeff(constr, slack_var, 1)
            elif sense == GRB.GREATER_EQUAL:
                model.chgCoeff(constr, slack_var, -1)

            # Set the constraint to equality after adding slack/surplus
            constr.Sense = GRB.EQUAL

        # Exclude the objective function variable from being part of the constraints
        if constr.ConstrName == 'obj':
            continue  # Skip any handling for the objective constraint

    model.update()

    return model

def construct_dual_model(standard_model):

    # Extract primal model data
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(standard_model)

    # Initialize the dual model
    dual_model = gp.Model('DualModel')

    # Number of primal constraints and variables
    num_constraints, num_variables = A.shape

    # Convert b and c to NumPy arrays if they're not already
    b = np.array(b)
    c = np.array(c)

    # Since all primal constraints are equality constraints, dual variables are unrestricted
    # Create dual variables as an MVar with unrestricted bounds
    dual_vars = dual_model.addMVar(shape=num_constraints, lb=-GRB.INFINITY, name='y')

    dual_model.update()

    # Set the dual objective function
    dual_obj = b @ dual_vars
    dual_obj += co  # Add constant term from primal objective if any

    # Determine dual objective sense and constraint direction based on primal objective sense
    if of_sense == GRB.MINIMIZE:
        # Primal is minimization: Dual is maximization
        dual_obj_sense = GRB.MAXIMIZE
        inequality_sense = '>='  # Dual constraints: A^T y >= c
    else:
        # Primal is maximization: Dual is minimization
        dual_obj_sense = GRB.MINIMIZE
        inequality_sense = '<='  # Dual constraints: A^T y <= c

    dual_model.setObjective(dual_obj, dual_obj_sense)

    # Compute the dual constraints
    # Transpose of A
    if isinstance(A, np.ndarray):
        A_transpose = A.T
    else:
        A_dense = A.toarray()
        A_transpose = A_dense.T

    # Compute constraint expressions
    constr_exprs = A_transpose @ dual_vars

    # Add dual constraints: A^T y >= c or A^T y <= c based on the primal objective sense
    for j in range(num_variables):
        constr_name = f'constraint_{variable_names[j]}'
        # if inequality_sense == '>=':
        #     dual_model.addConstr(constr_exprs[j] >= c[j], name=constr_name)
        # else:
        dual_model.addConstr(constr_exprs[j] <= c[j], name=constr_name)

    # No need to call update() after addConstr in recent Gurobi versions
    dual_model.update()
    return dual_model
