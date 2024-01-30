import numpy as np
from gurobipy import GRB
from scipy.sparse import csr_matrix
from Interpretable_Optimization.models.utils_models.utils_functions import get_model_matrices, save_json, \
    build_model_from_json


def get_row_activities(model):
    """
    Compute and return the support, minimal activity, and maximal activity for each row in a Gurobi model.

    Parameters:
    model (gurobipy.Model): The Gurobi model containing the optimization problem.

    Returns:
    support, min_activity, max_activity.
    SUPP - support: Set of indices j where a_ij is non-zero.
    INF - min_activity: Infimum of the row activity calculated as the sum of a_ij*l_j (for a_ij > 0) and a_ij*u_j (for a_ij < 0).
    SUP - max_activity: Supremum of the row activity calculated as the sum of a_ij*u_j (for a_ij > 0) and a_ij*l_j (for a_ij < 0).
    """

    rows = model.getConstrs()
    cols = model.getVars()
    SUPP = []
    SUP = []
    INF = []

    for i, row in enumerate(rows):
        support = set()
        min_activity = 0
        max_activity = 0

        for j, var in enumerate(cols):
            a_ij = model.getCoeff(row, var)
            l_j = var.LB
            u_j = var.UB

            if a_ij != 0:
                support.add(j)

                if a_ij > 0:
                    min_activity += a_ij * l_j
                    max_activity += a_ij * u_j
                else:
                    min_activity += a_ij * u_j
                    max_activity += a_ij * l_j

        SUPP.append(support)
        INF.append(min_activity)
        SUP.append(max_activity)

    return SUPP, INF, SUP


def feedback_individual_constraints(model, feasibility_tolerance=1e-6, infinity=1e30):
    """
    Analyzes each constraint in a Gurobi model and categorizes them as valid, redundant, or infeasible.

    Parameters:
    model (gurobipy.Model): The Gurobi model containing the optimization problem.
    feasibility_tolerance (float, optional): Tolerance used to assess feasibility. Defaults to 1e-6.
    infinity (float, optional): Value representing infinity in the context of the model. Defaults to 1e30.

    Returns:
    np.array: A matrix with each row containing the constraint number and its feedback ('valid', 'redundant', 'infeasible').
    """

    # Copy the model
    model_copy = model.copy()

    # Transform all '>=' constraints to '<=' in the copied model
    for constr in model_copy.getConstrs():
        if constr.Sense == '>':
            constr.Sense = '<='
            constr.RHS = -constr.RHS
            for i, var in enumerate(model_copy.getVars()):
                coeff = model_copy.getCoeff(constr, var)
                model_copy.chgCoeff(constr, var, -coeff)

    model_copy.update()
    # Get row activities from the model
    SUPP, INF, SUP = get_row_activities(model_copy)

    feedback_matrix = []

    for i, constr in enumerate(model_copy.getConstrs()):
        # Extracting constraint details
        sense = constr.Sense
        rhs = constr.RHS

        # Determining the feedback
        if sense == '<':
            if rhs >= infinity:
                feedback = 'redundant'
            elif SUP[i] <= rhs + feasibility_tolerance:
                feedback = 'redundant'
            elif INF[i] > rhs + feasibility_tolerance:
                feedback = 'infeasible'
            else:
                feedback = 'valid'
        elif sense == '=':
            if INF[i] >= rhs - feasibility_tolerance and SUP[i] <= rhs + feasibility_tolerance:
                feedback = 'redundant'
            elif INF[i] > rhs + feasibility_tolerance or SUP[i] < rhs - feasibility_tolerance:
                feedback = 'infeasible'
            else:
                feedback = 'valid'
        else:
            raise ValueError("Unsupported constraint sense: {}".format(sense))

        feedback_matrix.append([i, feedback])

    return np.array(feedback_matrix)


def small_coefficient_reduction(old_model, feasibility_tolerance=1e-6):
    """
    Perform pre-solve reduction on small coefficients in the constraint matrix of a Gurobi model.

    The function applies three reduction cases on the coefficients:
    1. Negligibly Small Coefficient: Coefficients smaller than 1e-3 meeting a specific product criterion are set to zero
     after adjusting the RHS.
    2. Cumulative Small Coefficient Reduction: Starting with the smallest coefficient in a constraint, coefficients are
    set to zero as long as the cumulative sum of changes remains below 10e-1 * feasibility_tolerance.
    3. Extremely Small Coefficient: Coefficients smaller than 1e-10 are directly set to zero.

    Parameters:
    old_model (gurobipy.Model): The Gurobi model to be modified.

    Returns:
    tuple: A tuple containing the updated model and a dictionary of changes.
           The dictionary keys are tuples (i, j) indicating the row and column indices,
           and values are dictionaries with keys 'old_value', 'new_value', and 'description'.
    """

    model = old_model.copy()
    changes = {}  # Track changes made to coefficients

    # Iterate through the constraints and variables to apply reductions
    for constr in model.getConstrs():
        i = constr.ConstrName
        expr = model.getRow(constr)  # Get variables and coefficients
        cumulative_modifications = 0  # Track cumulative sum of modifications for Case 2

        num_terms = expr.size()

        # Iterate over each term in the linear expression
        for k in range(num_terms):
            var = expr.getVar(k)
            coeff = expr.getCoeff(k)
            j = var.VarName

            # Case 1: Extremely Small Coefficient
            if abs(coeff) < 1e-10:
                changes[(i, j)] = {'old_value': coeff, 'new_value': 0, 'description': 'Extremely Small Coefficient'}
                # Set coefficient to zero in the model
                model.chgCoeff(constr, var, 0)
                cumulative_modifications += abs(coeff) * (var.UB - var.LB)

            # Case 2: Negligibly Small Coefficient
            elif abs(coeff) < 1e-3:
                product = abs(coeff) * (var.UB - var.LB) * abs(len(expr.getVars()))
                if product < 1e-2 * feasibility_tolerance:
                    changes[(i, j)] = {'old_value': coeff, 'new_value': 0,
                                       'description': 'Negligibly Small Coefficient'}
                    # Update RHS and set coefficient to zero in the model
                    constr.setAttr("RHS", constr.RHS - coeff * var.LB)
                    model.chgCoeff(constr, var, 0)
                    cumulative_modifications += abs(coeff) * (var.UB - var.LB)

    model.update()  # Update the model with the changes
    return model, changes


def eliminate_zero_rows(model):
    """
    Eliminate zero rows from a Gurobi optimization model.

    This function first creates a copy of the given model. It then transforms all
    '>=' constraints to '<=' constraints. The function checks each constraint to
    identify zero rows, i.e., rows where all coefficients are zero.

    Based on the right-hand side of these zero rows, the function classifies
    each constraint as:
    - 'Redundant' if the right-hand side is non-negative
    - 'Infeasible' if the right-hand side is negative
    - 'Valid' for all other constraints

    Parameters:
    - model: The Gurobi model to be processed.

    Returns:
    - A dictionary where each key is a constraint name, and the value is
      a string indicating whether the constraint is 'Redundant', 'Infeasible', or 'Valid'.
    """
    # Step 1: Copy the model
    model_copy = model.copy()

    # Step 2: Transform all '>=' constraints to '<=' in the copied model
    for constr in model_copy.getConstrs():
        if constr.Sense == '>':
            constr.Sense = '<='
            constr.RHS = -constr.RHS
            for i, var in enumerate(model_copy.getVars()):
                coeff = model_copy.getCoeff(constr, var)
                model_copy.chgCoeff(constr, var, -coeff)

    model_copy.update()

    # Step 3 & 4: Identify zero rows and classify constraints
    feedback = {}
    for constr in model_copy.getConstrs():
        is_zero_row = all(model_copy.getCoeff(constr, var) == 0 for var in model_copy.getVars())

        if is_zero_row:
            if constr.RHS >= 0:
                feedback[constr.ConstrName] = 'Redundant'
            else:
                feedback[constr.ConstrName] = 'Infeasible'
        else:
            feedback[constr.ConstrName] = 'Valid'

    return feedback


def eliminate_zero_columns(model):
    """
    This function evaluates each decision variable in a given Gurobi model to identify redundant or unbounded variables.
    It checks if a column in the coefficient matrix A is empty (all coefficients are zero).
    Depending on the cost coefficient c_j, the variable is classified as redundant, unbounded, or valid.

    Args:
    model (gurobipy.Model): The Gurobi model to evaluate.

    Returns:
    dict: A dictionary where keys are variable names and values are feedback strings ('Redundant', 'Unbounded', 'Valid').
    """

    feedback = {}
    A = model.getA()
    for j, var in enumerate(model.getVars()):
        # Extract the column corresponding to the variable
        col = A.A[:, j]

        # Check if all coefficients in the column are zero
        if all(c == 0 for c in col):
            c_j = var.obj

            # Classify based on c_j value
            if c_j >= 0:
                feedback[var.varName] = 'Redundant'
            else:
                feedback[var.varName] = 'Unbounded'
        else:
            feedback[var.varName] = 'Valid'

    return feedback


def eliminate_singleton_equalities(model, current_matrices_path):
    """
    This function processes a Gurobi model to eliminate singleton equality constraints.
    It simplifies the model by fixing variables and updating the model accordingly.
    If a negative fixed value for a variable is found, the model is declared infeasible.

    Args:
    model (gurobipy.Model): The Gurobi model to process.

    Returns:
    gurobipy.Model: The updated Gurobi model after eliminating singleton equalities.
    """

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Variable to track if we found a singleton in the current iteration
    found_singleton = True

    # Dictionary to store solutions for singletons
    solution = {}

    while found_singleton:
        found_singleton = False

        # Getting the matrices of the model
        A, b, c, lb, ub, of_sense, cons_senses = get_model_matrices(copied_model)

        # Getting objective function expression
        of = copied_model.getObjective()

        # Getting the variable names
        variable_names = [var.VarName for var in copied_model.getVars()]

        # Getting the number of nonzero elements per row
        nonzero_count_per_row = np.count_nonzero(A.A, axis=1)

        # Create a boolean array where True indicates rows with a single non-zero element
        single_nonzero = nonzero_count_per_row == 1

        # Create a boolean array where True indicates rows with '=' constraint sense
        equality_constraints = np.array(cons_senses) == '='

        # Combine the two conditions
        valid_rows = np.logical_and(single_nonzero, equality_constraints)

        # Find the index of the first row that satisfies both conditions
        first_singleton_index = np.where(valid_rows)[0][0] if np.any(valid_rows) else None

        # Check if the equality singleton row exists and process it
        if first_singleton_index is not None:

            found_singleton = True
            # Extract the singleton row
            singleton_row = A.A[first_singleton_index, :]

            # Identify the non-zero column (k)
            k_index = np.nonzero(singleton_row)[0][0]

            # Calculate the value for the variable corresponding to the singleton row
            x_k = b[first_singleton_index] / A.A[first_singleton_index, k_index]

            if x_k >= 0:
                # Update the solution dictionary
                solution[variable_names[k_index]] = x_k

                # update b
                b = b - A.A[:, k_index] * x_k
                b = np.delete(b, first_singleton_index)
                b = b.tolist()

                # update objective function constant
                if c[k_index] != 0:
                    of = of - c[k_index]*x_k

                # Update A
                A_new = np.delete(A.A, first_singleton_index, axis=0)  # Delete row
                A_new = np.delete(A_new, k_index, axis=1)  # Delete column
                A = csr_matrix(A_new)

                # update c, lb, ub and cons_senses
                del c[k_index]
                lb = np.delete(lb, k_index)
                ub = np.delete(ub, k_index)
                del cons_senses[first_singleton_index]
                del variable_names[k_index]

                save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, variable_names)
                copied_model = build_model_from_json(current_matrices_path)

            else:
                # Problem is infeasible
                return "Warning: Model is infeasible due to a negative singleton."

    # Update the model
    copied_model.update()
    return copied_model, solution
