import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os

# PATHS TO THE DATA
project_root = os.path.dirname(os.getcwd())
project_root = os.path.dirname(project_root)
model_path = os.path.join(project_root, 'data/GAMS_library', 'PDI.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'DINAM.mps')
real_model_path = os.path.join(project_root, 'data/real_data', 'openTEPES_EAPP_2030_sc01_st1.mps')


def canonical_model(original_model):
    """
    Convert a given model to standard form by eliminating free variables, handling finite bounds, and transforming
    inequality constraints into equalities.
    The probem in to minimize the objective function.
    Args:
        original_model (gurobipy.Model): The original model to convert to standard form.
    Returns:
        model (gurobipy.Model): The model in standard form.
    """
    # Create a copy of the original model to work on
    model = original_model.copy()
 
    # Step 1: Ensure the model is in minimization form
    if model.ModelSense != GRB.MINIMIZE:
        print("Model sense is not minimization")
        model.setObjective(-model.getObjective(), GRB.MINIMIZE)
 
    model.update()
    var_new_dict = {}
    new_vars = []
    obj_expr = gp.LinExpr()
    # Step 2: Handle free variables (convert to two non-negative variables)
    for var in model.getVars():
        lb = var.LB
        ub = var.UB
        obj_coeff = var.Obj
        vtype = var.VType
        # Case 1: Variable is free (lb = -infinity, ub = infinity)
        if lb == float('-inf') and ub == float('inf'):
            # Decompose into two non-negative variables: var = var_pos - var_neg
            var_pos = model.addVar(lb=0, ub=GRB.INFINITY, vtype=vtype, name=f"{var.VarName}_pos")
            var_neg = model.addVar(lb=0, ub=GRB.INFINITY, vtype=vtype, name=f"{var.VarName}_neg")
            var_new_dict[var] = (var_pos, var_neg)
            new_vars.extend([var_pos, var_neg])

            # Adjust the objective function
            if obj_coeff != 0:
                obj_expr.addTerms(obj_coeff, var_pos)
                obj_expr.addTerms(-obj_coeff, var_neg)

        # Case 2: Variable has lb = -infinity, ub finite
        elif (lb == float('-inf')) and (ub < GRB.INFINITY):
            # Represent var as var = ub - y, y >= 0
            y = model.addVar(lb=0, ub=GRB.INFINITY, vtype=vtype, name=f"{var.VarName}_shifted")
            var_new_dict[var] = y
            new_vars.append(y)

            # Adjust the objective function
            if obj_coeff != 0:
                obj_expr.addConstant(obj_coeff * ub)
                obj_expr.addTerms(-obj_coeff, y)

        # Case 3: Variable has finite lb, ub = infinity
        elif lb > float('-inf') and ub == float('inf'):
            # Shift variable to have lb = 0: var = y + lb
            y = model.addVar(lb=0, ub=GRB.INFINITY, vtype=vtype, name=f"{var.VarName}_shifted")
            var_new_dict[var] = y
            new_vars.append(y)

            # Adjust the objective function
            if obj_coeff != 0:
                obj_expr.addConstant(obj_coeff * lb)
                obj_expr.addTerms(obj_coeff, y)

        # Case 4: Variable has finite lb and ub
        elif lb > float('-inf') and ub < GRB.INFINITY:
            # Shift variable to have lb = 0: var = y + lb, y <= ub - lb
            y = model.addVar(lb=0, ub=GRB.INFINITY, vtype=vtype, name=f"{var.VarName}_shifted")
            var_new_dict[var] = y
            new_vars.append(y)
            # Add a constraint to enforce the upper bound
            model.addConstr(y <= ub - lb, name=f"{var.VarName}_UB")
            

            # Adjust the objective function
            if obj_coeff != 0:
                obj_expr.addConstant(obj_coeff * lb)
                obj_expr.addTerms(obj_coeff, y)
        else:
            # Should not reach here
            raise ValueError(f"Unexpected variable bounds for variable {var.VarName}: LB={lb}, UB={ub}")
        
    # Add independent terms in the objective function
    obj_expr.addConstant(model.getObjective().getConstant())


    model.update()

    # Adjust constraints
    for constr in model.getConstrs():
        row = model.getRow(constr)
        new_expr = gp.LinExpr()
        constant_term = 0

        for idx in range(row.size()):
            var = row.getVar(idx)
            coeff = row.getCoeff(idx)

            if var in var_new_dict:
                new_var = var_new_dict[var]
                if isinstance(new_var, tuple):
                    # var = var_pos - var_neg
                    var_pos, var_neg = new_var
                    new_expr.addTerms(coeff, var_pos)
                    new_expr.addTerms(-coeff, var_neg)
                elif var.LB == float('-inf') and var.UB < GRB.INFINITY:
                    # var = ub - y
                    y = new_var
                    constant_term += coeff * var.UB
                    new_expr.addTerms(-coeff, y)
                elif var.LB > -GRB.INFINITY and var.UB == float('inf'):
                    # var = y + lb
                    y = new_var
                    constant_term += coeff * var.LB
                    new_expr.addTerms(coeff, y)
                elif var.LB > -GRB.INFINITY and var.UB < GRB.INFINITY:
                    # var = y + lb
                    y = new_var
                    constant_term += coeff * var.LB
                    new_expr.addTerms(coeff, y)
                else:
                    raise ValueError("Unhandled variable case in constraints")
            else:
                new_expr.addTerms(coeff, var)

        # Adjust the RHS of the constraint
        sense = constr.Sense
        rhs = constr.RHS
        if constant_term != 0:
            # Move constant term to RHS
            rhs -= constant_term

        # Replace the constraint with the new expression
        model.remove(constr)
        # Ensure `sense` is correctly mapped to a comparison operator
        if sense == GRB.LESS_EQUAL:
            model.addConstr(new_expr <= rhs, name=constr.ConstrName)
        elif sense == GRB.GREATER_EQUAL:
            model.addConstr(new_expr >= rhs, name=constr.ConstrName)
        elif sense == GRB.EQUAL:
            model.addConstr(new_expr == rhs, name=constr.ConstrName)
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")

    model.update()

    # Set the new objective function
    #obj_expr.addConstant(model.getObjective().getConstant())
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Remove old variables
    model.remove(list(var_new_dict.keys()))

    # Update the model
    model.update()

    # Step 4: Transform all constraints into less than or equal to constraints
    equality_constrs = [constri for constri in model.getConstrs() if constri.Sense == GRB.EQUAL]

    for constr in equality_constrs:
        row = model.getRow(constr)
        model.remove(constr)
        model.addConstr(row <= constr.RHS, name=constr.ConstrName)
        model.addConstr(row >= constr.RHS, name=f"{constr.ConstrName}_greater")

    model.update()
    greater_constrs = [constri for constri in model.getConstrs() if constri.Sense == GRB.GREATER_EQUAL]
    for constr in greater_constrs:
        row = model.getRow(constr)
        model.remove(constr)
        model.addConstr(-row <= -constr.RHS, name=constr.ConstrName)
    
    model.update()
    return model 


if __name__ == '__main__': 
    # Load the model
    original_model = gp.read(real_model_path)
    original_model.setParam('OutputFlag', 0)
    original_model.optimize()
    print("Original objective value:", original_model.objVal)
    # Convert the model to standard form
    model = canonical_model(original_model)
    model.setParam('OutputFlag', 0)
    model.optimize()
    print("Optimal objective value:", model.objVal)