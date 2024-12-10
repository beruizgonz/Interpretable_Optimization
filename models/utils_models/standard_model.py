import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils_models.utils_functions import *
import gurobipy as gp
from gurobipy import GRB

def standard_form_e1(original_model):
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
        model.addConstr(new_expr, sense, rhs, name=constr.ConstrName)

    model.update()

    # Set the new objective function
    #obj_expr.addConstant(model.getObjective().getConstant())
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Remove old variables
    model.remove(list(var_new_dict.keys()))

    # Update the model
    model.update()

    # Step 4: Transform all constraints into equalities by introducing slack/surplus variables
    ineq_constrs = [constr for constr in model.getConstrs() if constr.Sense != GRB.EQUAL]

    # Add slack/surplus variables for all inequality constraints
    slack_vars = []
    for constr in ineq_constrs:
        # Print statement for debugging purposes (can be removed in production code)
        #print(f"Constraint: {constr.ConstrName}, Sense: {constr.Sense}")
        slack_var = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"slack_{constr.ConstrName}")
        slack_vars.append((constr, slack_var))
    
    # Update the model once after adding all slack variables
    model.update()
    
    # Adjust constraints to include slack variables and change to equality
    for constr, slack_var in slack_vars:
        if constr.Sense == GRB.LESS_EQUAL:
            model.chgCoeff(constr, slack_var, 1)
        elif constr.Sense == GRB.GREATER_EQUAL:
            model.chgCoeff(constr, slack_var, -1)
        # Change the constraint's sense to equality
        constr.Sense = GRB.EQUAL
    
    # Final model update
    model.update()
    return model

def standard_form1(originañ_model):

    model = originañ_model.copy()
    # Step 1: Ensure the model is in minimization form
    if model.ModelSense != 1:
        model.setObjective(-1 * model.getObjective(), GRB.MINIMIZE)
        model.ModelSense = 1  # Set the model sense to minimization

    model.update()
    # Step 2: Ensure all variables are non-negative
    for var in model.getVars():
        if var.LB != 0 or var.UB < GRB.INFINITY or var.LB == -GRB.INFINITY:
            # Create a new variable with lb=0 and no upper bound
            var_new = model.addVar(lb=-GRB.INFINITY, vtype=var.VType, name=f"{var.VarName}_new")
            model.update()

            # Replace 'var' in constraints with 'var_new' and enforce the original bounds
            for constr in model.getConstrs():
                coeff = model.getCoeff(constr, var)
                if coeff != 0:
                    model.chgCoeff(constr, var_new, coeff)
                    #model.chgCoeff(constr, var, 0)

            # Add constraints to enforce original bounds if needed
            if var.LB > -GRB.INFINITY and var.LB != 0:
                    model.addConstr(var_new >= var.LB, name=f"{var.VarName}_LB")
            if var.UB < GRB.INFINITY:
                    model.addConstr(var_new <= var.UB, name=f"{var.VarName}_UB")

            # Replace 'var' in the objective function
            obj_coeff = var.getAttr(GRB.Attr.Obj)
            if obj_coeff != 0:
                model.setObjective(model.getObjective() + obj_coeff * var_new, GRB.MINIMIZE)

            # Remove the original variable from the model
            model.remove(var)

    model.update()

    # Step 3: Transform all constraints into equalities by introducing slack/surplus variables
    for constr in model.getConstrs():
        sense = constr.Sense
        if constr.ConstrName == 'obj':
            continue  # Skip any handling for the objective constraint

        if sense != GRB.EQUAL:
            # Add slack or surplus variable based on constraint type
            slack_var = model.addVar(lb=0, name=f"slack_{constr.ConstrName}")
            model.update()

            # Add slack or surplus depending on the sense of the constraint
            if sense == GRB.LESS_EQUAL:
                model.chgCoeff(constr, slack_var, 1)
            if sense == GRB.GREATER_EQUAL:
                model.chgCoeff(constr, slack_var, -1)
            # Set the constraint to equality after adding slack/surplus
            constr.Sense = GRB.EQUAL

    model.update()

    return model

def standard_form2(model): 
    # Step 1: Ensure the model is in minimization form 
    if model.ModelSense != 1: 
        model.setObjective(-1 * model.getObjective(), GRB.MINIMIZE) 
        model.ModelSense = 1  # Set the model sense to minimization 
 
    # Step 2: Ensure all variables are non-negative 
    for var in model.getVars():
        if var.LB < 0: 
            # Replace variable with two non-negative variables
            pos_var = model.addVar(lb=0, vtype=var.VType, name=f"{var.VarName}_pos")
            neg_var = model.addVar(lb=0, vtype=var.VType, name=f"{var.VarName}_neg")
            # Update constraints to replace old variable with the new positive and negative parts
            for constr in model.getConstrs():
                coeff = model.getCoeff(constr, var)
                if coeff != 0:
                    model.chgCoeff(constr, pos_var, coeff)
                    model.chgCoeff(constr, neg_var, -coeff)
                    model.chgCoeff(constr, var, 0)

            # Update the objective function if the original variable was part of it
            obj_coeff = var.getAttr(GRB.Attr.Obj)
            if obj_coeff != 0:
                model.setObjective(model.getObjective() + obj_coeff * (pos_var - neg_var), GRB.MINIMIZE)

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


def construct_model_from_json(data_path):
    A, b, c, lb, ub, of_sense, cons_senses, co, variable_names = None, None, None, None, None, None, None, None, None

    # Load data from JSON files
    file_names = ['A.json', 'b.json', 'c.json', 'lb.json', 'ub.json', 'of_sense.json', 'cons_senses.json', 'co.json',
                  'variable_names.json']
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

                if file_name == 'A.json':
                    A = np.array(data)
                elif file_name == 'b.json':
                    b = np.array(data)
                elif file_name == 'c.json':
                    c = np.array(data)
                elif file_name == 'lb.json':
                    lb = np.array(data)
                elif file_name == 'ub.json':
                    ub = np.array(data)
                elif file_name == 'of_sense.json':
                    of_sense = data
                elif file_name == 'cons_senses.json':
                    cons_senses = data
                elif file_name == 'co.json':
                    co = data
                elif file_name == 'variable_names.json':
                    variable_names = data
        except FileNotFoundError:
            # Skip if the file does not exist
            continue

    # Create a Gurobi model and add variables
    num_variables = len(c)
    model = gp.Model()
    model.addMVar(num_variables, lb=lb, ub=ub, name=variable_names)
    model.update()

    b = np.array(b)
    c = np.array(c)

    # Set the objective function
    model.setObjective(c @ model.getVars()+ co, GRB.MINIMIZE)

    # Add constraints they are all equality constraints
    for i in range(A.shape[0]):
        model.addConstr(A[i] @ model.getVars() == b[i], name=f'constraint_{i}')
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
        A_dense = A
    else:
        A_dense = A.toarray()
        A_transpose = A_dense.T

    # Compute constraint expressions
    constr_exprs = dual_vars.T @ A_dense

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

if __name__ == '__main__':   

#The path to the data
    project_root = os.path.dirname(os.getcwd())
    project_root = os.path.dirname(project_root)
    model_path = os.path.join(project_root, 'data/GAMS_library', 'DINAM.mps')
    GAMS_path = os.path.join(project_root, 'data/GAMS_library')
    GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
    model_path_modified = os.path.join(GAMS_path_modified, 'DINAM.mps')
    model = gp.read(model_path_modified)
    standard_model = standard_form_e1(model)

    # model.optimize()
    # print(model.objVal)
    # standard = standard_form1(model)
    # standard.optimize()
    # print(standard.objVal)
    # # # Apply standard form 1
    # model_standard1 = standard_form1(model)
    # model_standard1.optimize()  
    # del model_standard1
    # model = gp.read(model_path_modified)
    # # Apply standard form 2
    # model_standard2 = standard_form2(model)
    # model_standard2.optimize()
    # print("Model standard 2 solved")
    # print(model_standard2.objVal)
