import numpy as np
import gurobipy as gp   
from gurobipy import GRB
from scipy.sparse import csr_matrix



def flexibility_constraints(model, constraint_names, vars_remove,M):
    """
    Efficiently determines the minimum and maximum coefficient values for each row 
    related to the variables in `vars_remove`, and updates constraint senses accordingly.

    Parameters:
        A (scipy.sparse matrix): The constraint matrix.
        b (numpy array): The right-hand side vector of the constraints.
        cons_senses (dict): Dictionary mapping row indices to constraint senses.
        vars_remove (dict): Dictionary mapping row indices to sets of column indices to remove.

    Returns:
        cons_senses (dict): Updated constraint senses.
        new_vars (set): Set of variables affected by constraint modifications.
    """    
    new_vars = []
    for row in vars_remove:
        constraint_name = constraint_names[row]  

        # Add excess variable with constraint-specific naming
        excess_var = model.addVar(
            lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, 
            name=f"excess_{constraint_name}"
        )
        new_vars.append(excess_var)
        # Add slack variable with constraint-specific naming
        slack_var = model.addVar(
            lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, 
            name=f"defect_{constraint_name}"
        )
        new_vars.append(slack_var)

    model.update()  
    # Modify constraints to include excess and slack variables
    for constr in model.getConstrs():
        constr_name = constr.ConstrName 
        excess_var = model.getVarByName(f"excess_{constr_name}")
        slack_var = model.getVarByName(f"defect_{constr_name}")
        # Add the new constraint
        if excess_var and slack_var:
            row_expr = model.getRow(constr)
            row_expr += excess_var - slack_var 

            model.chgCoeff(constr, excess_var, 1) 
            model.chgCoeff(constr, slack_var, -1)

    model.update()
    current_obj = model.getObjective()
    model.setObjective(current_obj + M*gp.quicksum(var for var in new_vars), GRB.MINIMIZE)
    model.update()
    return model


def eliminate_constraints(model, rows_to_delete):
    """
    Efficiently eliminates constraints from a model.

    Parameters:
        model (gurobipy.Model): The model to modify.
        rows_to_delete (set): Set of row indices to delete.

    Returns:
        model (gurobipy.Model): The modified model.
    """
    for row in rows_to_delete:
        constraint_name = model.getConstrs()[row].ConstrName
        model.remove(model.getConstrByName(constraint_name))
    model.update()
    return model

def eliminate_constraints_matrix(A, b, rows_to_delete): 
    """

    Efficiently eliminates constraints from a constraint matrix.
    Parameters:
        A (scipy.sparse matrix): The constraint matrix.
        b (numpy array): The right-hand side vector of the constraints.
        rows_to_delete (set): Set of row indices to delete.
    Returns:
        A (scipy.sparse matrix): The modified constraint matrix.
        b (numpy array): The modified right-hand side vector.
    """
    num_rows = A.shape[0]
    mask = np.ones(num_rows, dtype=bool)
    mask[rows_to_delete] = False
    A = csr_matrix(spdiags(mask.astype(int), 0, num_rows, num_rows).dot(A))
    b[rows_to_delete] = 0
    return A, b
