import numpy as np
from gurobipy import GRB
from scipy.sparse import csr_matrix
from Interpretable_Optimization.models.utils_models.utils_functions import get_model_matrices, save_json, \
    build_model_from_json
from collections import defaultdict

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


def eliminate_zero_rows(model, current_matrices_path):
    """
    Eliminate zero rows from a Gurobi optimization model.

    This function first creates a copy of the given model. It then checks each
    constraint to identify zero rows, i.e., rows where all coefficients are zero.

    Based on the right-hand side and the sense of these zero rows, the function
    classifies each constraint as:
    - 'Redundant' if the constraint is rendered redundant based on the rules
    - 'Infeasible' if the constraint renders the model infeasible
    - 'Valid' for all other constraints

    Parameters:
    - model: The Gurobi model to be processed.

    Returns:
    - A dictionary where each key is a constraint name, and the value is
      a string indicating whether the constraint is 'Redundant', 'Infeasible', or 'Valid'.
    """
    try:
        # Copy the model
        model_copy = model.copy()

        # Getting the matrices of the model
        A, b, c, co, lb, ub, of_sense, cons_senses = get_model_matrices(model_copy)

        # Getting objective function expression
        of = model_copy.getObjective()
        co = of.getConstant()

        # Getting the variable names
        variable_names = [var.VarName for var in model_copy.getVars()]

        # Identify zero rows and classify constraints
        feedback = {}
        to_delete = []
        for i, constr in enumerate(model_copy.getConstrs()):
            is_zero_row = all(model_copy.getCoeff(constr, var) == 0 for var in model_copy.getVars())

            if is_zero_row:
                if constr.Sense == GRB.LESS_EQUAL:
                    if constr.RHS >= 0:
                        feedback[constr.ConstrName] = 'Redundant'
                        to_delete.append(i)
                    else:
                        feedback[constr.ConstrName] = 'Infeasible'
                elif constr.Sense == GRB.GREATER_EQUAL:
                    if constr.RHS <= 0:
                        feedback[constr.ConstrName] = 'Redundant'
                        to_delete.append(i)
                    else:
                        feedback[constr.ConstrName] = 'Infeasible'
                elif constr.Sense == GRB.EQUAL:
                    if constr.RHS == 0:
                        feedback[constr.ConstrName] = 'Redundant'
                        to_delete.append(i)
                    else:
                        feedback[constr.ConstrName] = 'Infeasible'
            else:
                feedback[constr.ConstrName] = 'Valid'

        # update A and b excluding the zero-constraints
        A_new = np.delete(A.A, to_delete, axis=0)
        A = csr_matrix(A_new)

        # Delete elements from b
        for index in to_delete:
            del b[index]

        save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
        model_copy = build_model_from_json(current_matrices_path)

        return model_copy, feedback

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def eliminate_zero_columns(model, current_matrices_path):
    """
    This function evaluates each decision variable in a given Gurobi model to identify redundant or unbounded variables.
    It checks if a column in the coefficient matrix A is empty (all coefficients are zero).
    Depending on the cost coefficient c_j, the variable is classified as redundant, unbounded, or valid.

    Args:
    model (gurobipy.Model): The Gurobi model to evaluate.

    Returns:
    dict: A dictionary where keys are variable names and values are feedback strings ('Redundant', 'Unbounded', 'Valid').
    """

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Getting the matrices of the model
    A, b, c, co, lb, ub, of_sense, cons_senses = get_model_matrices(copied_model)

    # Getting objective function expression
    of = copied_model.getObjective()
    co = of.getConstant()

    # Getting the variable names
    variable_names = [var.VarName for var in copied_model.getVars()]

    feedback = {}
    to_delete = []

    for j, var in enumerate(model.getVars()):
        # Extract the column corresponding to the variable
        col = A.A[:, j]

        # Check if all coefficients in the column are zero
        if all(c == 0 for c in col):
            c_j = var.obj

            # Classify based on c_j value
            if c_j >= 0:
                feedback[var.varName] = 'Redundant'
                to_delete.append(j)
            else:
                feedback[var.varName] = 'Unbounded'
        else:
            feedback[var.varName] = 'Valid'

    # update A and c
    A_new = np.delete(A.A, to_delete, axis=1)  # Delete column
    A = csr_matrix(A_new)

    # Delete elements from b
    for index in to_delete:
        del c[index]
        del variable_names[index]

    save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
    copied_model = build_model_from_json(current_matrices_path)

    return copied_model, feedback


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
        A, b, c, co, lb, ub, of_sense, cons_senses = get_model_matrices(copied_model)

        # Getting objective function expression
        of = copied_model.getObjective()
        co = of.getConstant()

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
                    co -= co - c[k_index] * x_k

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

                save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
                copied_model = build_model_from_json(current_matrices_path)

            else:
                # Problem is infeasible
                return "Warning: Model is infeasible due to a negative singleton."

    # Update the model
    copied_model.update()
    return copied_model, solution


def eliminate_doubleton_equalities(model, current_matrices_path):
    """
    This function processes a Gurobi model to eliminate doubleton equality constraints.

    Args:
    model (gurobipy.Model): The Gurobi model to process.
    current_matrices_path (str): Path to store and retrieve updated matrices.

    Returns:
    gurobipy.Model: The updated Gurobi model after eliminating doubleton equalities.
    dict: Solution dictionary with variable values for doubletons.
    """

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Variable to track if we found a singleton in the current iteration
    found_doubleton = True

    while found_doubleton:
        found_doubleton = False

        # Getting the matrices of the model
        A, b, c, co, lb, ub, of_sense, cons_senses = get_model_matrices(copied_model)

        # Getting objective function expression
        of = copied_model.getObjective()
        co = of.getConstant()
        # Getting the variable names
        variable_names = [var.VarName for var in copied_model.getVars()]

        # Getting the number of nonzero elements per row
        nonzero_count_per_row = np.count_nonzero(A.A, axis=1)

        # Create a boolean array where True indicates rows with two non-zero element
        single_nonzero = nonzero_count_per_row == 2

        # Create a boolean array where True indicates rows with '=' constraint sense
        equality_constraints = np.array(cons_senses) == '='

        # Combine the two conditions
        valid_rows = np.logical_and(single_nonzero, equality_constraints)

        # Find indices where the element is True
        true_indices = np.where(valid_rows)[0]

        # Check if there are any True values and get the first index, else set to None
        first_doubleton_index = true_indices[0] if true_indices.size > 0 else None

        # Check if the equality doubleton row exists and process it
        if first_doubleton_index is not None:
            found_doubleton = True
            # Extract the doubleton row
            doubleton_row = A.A[first_doubleton_index, :]

            # Identify the index of the last non-zero column (k)
            doubleton_column = np.nonzero(doubleton_row)[0][-1]

            # Update the element "doubleton_row" of b
            A_new = A.A.copy()
            b[first_doubleton_index] = b[first_doubleton_index] / A_new[first_doubleton_index, doubleton_column]

            # Update the row "doubleton_row" of A
            A_new[first_doubleton_index, :] = A_new[first_doubleton_index, :] / A_new[
                first_doubleton_index, doubleton_column]

            # Update the remaining rows/elements of A and b
            for i in range(A_new.shape[0]):
                if i != first_doubleton_index and A_new[i, doubleton_column] != 0:
                    b[i] -= A_new[i, doubleton_column] * b[first_doubleton_index]
                    A_new[i, :] -= A_new[i, doubleton_column] * A_new[first_doubleton_index, :]

            # Update the objective function
            if c[doubleton_column] != 0:
                co += c[doubleton_column] * b[first_doubleton_index]

            # Update c
            if c[doubleton_column] != 0:
                c -= c[doubleton_column]*A_new[first_doubleton_index, :]
            c = c.tolist()

            # Update values by eliminating the variable "doubleton_column"
            A_new = np.delete(A_new, doubleton_column, axis=1)  # Delete column
            A = csr_matrix(A_new)
            del c[doubleton_column]
            lb = np.delete(lb, doubleton_column)
            ub = np.delete(ub, doubleton_column)
            del variable_names[doubleton_column]

            cons_senses[first_doubleton_index] = '<='

            save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
            copied_model = build_model_from_json(current_matrices_path)

    copied_model.update()
    return copied_model


def nested_dict():
    return defaultdict(nested_dict)


def eliminate_kton_equalities(model, current_matrices_path, k):
    """
    This function processes a Gurobi model to eliminate k-tons equality constraints.

    Args:
    model (gurobipy.Model): The Gurobi model to process.
    current_matrices_path (str): Path to store and retrieve updated matrices.

    Returns:
    gurobipy.Model: The updated Gurobi model after eliminating doubleton equalities.
    dict: Solution dictionary with variable values for doubletons.
    """
    # iteration
    it = 0

    # Dictionary to store k-ton rows and their original equations
    kton_dict = defaultdict(nested_dict)

    # adjustment on the input data k
    A = model.getA()
    if k>(A.A.shape[1]-1):
        k = A.A.shape[1]-1

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    while k>1:
        copied_model.update()
        # Variable to track if we found a k-ton in the current iteration
        found_kton = True

        while found_kton:

            found_kton = False

            # Getting the matrices of the model
            A, b, c, co, lb, ub, of_sense, cons_senses = get_model_matrices(copied_model)

            # # Getting objective function expression
            # of = copied_model.getObjective()

            # Getting the variable names
            variable_names = [var.VarName for var in copied_model.getVars()]

            # Getting the number of nonzero elements per row
            nonzero_count_per_row = np.count_nonzero(A.A, axis=1)

            # Create a boolean array where True indicates rows with k non-zero element
            single_nonzero = nonzero_count_per_row == k

            # Create a boolean array where True indicates rows with '=' constraint sense
            equality_constraints = np.array(cons_senses) == '='

            # Combine the two conditions
            valid_rows = np.logical_and(single_nonzero, equality_constraints)

            # Find indices where the element is True
            true_indices = np.where(valid_rows)[0]

            # Check if there are any True values and get the first index, else set to None
            first_kton_index = true_indices[0] if true_indices.size > 0 else None

            # Check if the equality doubleton row exists and process it
            if first_kton_index is not None:
                found_kton = True

                it += 1

                # Extract the kton row
                kton_row = A.A[first_kton_index, :]

                # Identify the index of the last non-zero column (k)
                kton_column = np.nonzero(kton_row)[0][-1]

                # Save the k-ton constraint details
                kton_constraint = copied_model.getConstrs()[first_kton_index]
                constraint_key = f"iteration={it}, k-ton={k}"
                kton_dict[constraint_key] = {
                    'lfs': kton_row.copy(),
                    'vars': variable_names.copy(),
                    'rhs': b[first_kton_index],
                    'removed_var': variable_names[kton_column]
                }

                # Update the element "kton_row" of b
                A_new = A.A.copy()
                b[first_kton_index] = b[first_kton_index] / A_new[first_kton_index, kton_column]

                # Update the row "kton_row" of A
                A_new[first_kton_index, :] = A_new[first_kton_index, :] / A_new[first_kton_index, kton_column]

                # Update the remaining rows/elements of A and b
                for i in range(A_new.shape[0]):
                    if i != first_kton_index:
                        b[i] -= A_new[i, kton_column] * b[first_kton_index]
                        A_new[i, :] -= A_new[i, kton_column] * A_new[first_kton_index, :]

                # Update the objective function
                if c[kton_column] != 0:
                    co += c[kton_column] * b[first_kton_index]

                # Update c
                if c[kton_column] != 0:
                    c -= c[kton_column]*A_new[first_kton_index, :]
                c = c.tolist()

                # Update values by eliminating the variable "k_column"
                A_new = np.delete(A_new, kton_column, axis=1)  # Delete column
                A = csr_matrix(A_new)
                del c[kton_column]
                lb = np.delete(lb, kton_column)
                ub = np.delete(ub, kton_column)
                del variable_names[kton_column]

                cons_senses[first_kton_index] = '<='

                save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
                copied_model = build_model_from_json(current_matrices_path)
                copied_model.update()
            else:
                k -=1
    copied_model.update()
    return copied_model, kton_dict
