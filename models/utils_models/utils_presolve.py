import numpy as np
from scipy.sparse import csr_matrix
from Interpretable_Optimization.models.utils_models.utils_functions import get_model_matrices, save_json, \
    build_model_from_json, find_corresponding_negative_rows_with_indices, canonical_form, linear_dependency
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


def eliminate_implied_bounds(model, current_matrices_path, feasibility_tolerance=1e-6, infinity=1e30):
    """
    Analyzes each constraint in a Gurobi model and categorizes them as valid, redundant, or infeasible.

    Parameters:
    model (gurobipy.Model): The Gurobi model containing the optimization problem.
    feasibility_tolerance (float, optional): Tolerance used to assess feasibility. Defaults to 1e-6.
    infinity (float, optional): Value representing infinity in the context of the model. Defaults to 1e30.

    Returns:
    np.array: A matrix with each row containing the constraint number and its feedback ('valid', 'redundant', 'infeasible').
    """

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Getting the matrices of the model
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

    # Getting the variable names
    variable_names = [var.VarName for var in copied_model.getVars()]

    # Get row activities from the model
    SUPP, INF, SUP = get_row_activities(copied_model)

    feedback_constraint = {i: 'Valid' for i in range(len(copied_model.getConstrs()))}
    to_delete_constraint = []

    for i, constr in enumerate(copied_model.getConstrs()):

        if constr.RHS >= infinity:
            feedback_constraint[i] = 'Redundant'
            to_delete_constraint.append(i)

        if INF[i] > (constr.RHS + feasibility_tolerance):
            feedback_constraint[i] = 'Redundant'
            to_delete_constraint.append(i)

        if SUP[i] < (constr.RHS + feasibility_tolerance):
            feedback_constraint[i] = 'infeasible'

    # Exclude constraints
    A_new = np.delete(A.A, to_delete_constraint, axis=0)  # remove constraint (row)
    A = csr_matrix(A_new)

    # Delete elements from b and the corresponding elements of the constraint operation
    # Iterating in reverse order to avoid index shifting issues
    for index in sorted(to_delete_constraint, reverse=True):
        del b[index]
        del cons_senses[index]

    save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
    copied_model = build_model_from_json(current_matrices_path)
    copied_model.update()

    return copied_model, feedback_constraint


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
        A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model_copy)

        # Getting the variable names
        variable_names = [var.VarName for var in model_copy.getVars()]

        # Identify zero rows and classify constraints
        feedback = {}
        to_delete = []
        for i, constr in enumerate(model_copy.getConstrs()):
            is_zero_row = all(model_copy.getCoeff(constr, var) == 0 for var in model_copy.getVars())

            if is_zero_row:
                if constr.RHS <= 0:
                    feedback[constr.ConstrName] = 'Redundant'
                    to_delete.append(i)
                else:
                    feedback[constr.ConstrName] = 'Infeasible'
            else:
                feedback[constr.ConstrName] = 'Valid'

        # Removing columns of A using the list to_delete
        A_new = np.delete(A.A, to_delete, axis=0)  # Note: axis=1 for columns
        A = csr_matrix(A_new)

        # Delete elements from b and the constraint senses
        # Iterating in reverse order to avoid index shifting issues
        for index in sorted(to_delete, reverse=True):
            del b[index]
            del cons_senses[index]

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
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

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

    # Delete elements from c, variable_names, lb, and ub
    # Iterating in reverse order to avoid index shifting issues
    for index in sorted(to_delete, reverse=True):
        del c[index]
        del variable_names[index]
        lb = np.delete(lb, index)
        ub = np.delete(ub, index)

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
        A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

        # Getting the variable names
        variable_names = [var.VarName for var in copied_model.getVars()]

        # Getting the number of nonzero elements per row
        nonzero_count_per_row = np.count_nonzero(A.A, axis=1)

        # Create a boolean array where True indicates rows with a single non-zero element
        single_nonzero = nonzero_count_per_row == 1

        has_negative_counterpart, indices_list = find_corresponding_negative_rows_with_indices(A, b)

        # Combine the two conditions -> single nonzero + negative counterpart
        valid_rows = np.logical_and(single_nonzero, has_negative_counterpart)

        # Find the index of the first row that satisfies both conditions
        first_singleton_index = np.where(valid_rows)[0][0] if np.any(valid_rows) else None

        # Check if the equality singleton row exists and process it
        if first_singleton_index is not None:

            found_singleton = True
            # Extract the singleton row
            singleton_row = A.A[first_singleton_index, :]

            # identifying rows to delete
            index_rows_to_delete = indices_list[first_singleton_index]

            # Identify the non-zero column (k)
            k_index = np.nonzero(singleton_row)[0][0]

            # Calculate the value for the variable corresponding to the singleton row
            x_k = b[first_singleton_index] / A.A[first_singleton_index, k_index]

            if x_k >= 0:
                # Update the solution dictionary
                solution[variable_names[k_index]] = x_k

                # update b
                b = b - A.A[:, k_index] * x_k
                b = b.tolist()

                # Delete elements from b and the corresponding elements of the constraint operation
                # Iterating in reverse order to avoid index shifting issues
                for index in sorted(index_rows_to_delete, reverse=True):
                    del b[index]
                    del cons_senses[index]

                # update objective function constant
                if c[k_index] != 0:
                    co -= co - c[k_index] * x_k

                # Update A
                A_new = np.delete(A.A, index_rows_to_delete, axis=0)  # Delete rows
                A_new = np.delete(A_new, k_index, axis=1)  # Delete column
                A = csr_matrix(A_new)

                # update c, lb, ub and cons_senses
                del c[k_index]
                lb = np.delete(lb, k_index)
                ub = np.delete(ub, k_index)
                del variable_names[k_index]

                save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
                copied_model = build_model_from_json(current_matrices_path)
                copied_model.update()

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
        A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

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
                c -= c[doubleton_column] * A_new[first_doubleton_index, :]
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
    if k > (A.A.shape[1] - 1):
        k = A.A.shape[1] - 1

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    while k > 1:
        copied_model.update()
        # Variable to track if we found a k-ton in the current iteration
        found_kton = True

        while found_kton:

            found_kton = False

            # Getting the matrices of the model
            A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

            # Getting the variable names
            variable_names = [var.VarName for var in copied_model.getVars()]

            # Getting the number of nonzero elements per row
            nonzero_count_per_row = np.count_nonzero(A.A, axis=1)

            # Create a boolean array where True indicates rows with k non-zero element
            k_nonzero = nonzero_count_per_row == k

            has_negative_counterpart, indices_list = find_corresponding_negative_rows_with_indices(A, b)

            # Combine the two conditions
            valid_rows = np.logical_and(k_nonzero, has_negative_counterpart)

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

                # identifying rows to delete
                index_rows_to_delete = indices_list[first_kton_index]

                # Identify the index of the last non-zero column (k)
                kton_column = np.nonzero(kton_row)[0][-1]

                # Save the k-ton constraint details

                constraint_key = f"iteration={it}, k-ton={k}"
                kton_dict[constraint_key] = {
                    'lhs': kton_row.copy(),
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
                co += c[kton_column] * b[first_kton_index]

                # Update c
                c -= c[kton_column] * A_new[first_kton_index, :]
                c = c.tolist()

                # Update values by eliminating the variable "k_column"
                A_new = np.delete(A_new, kton_column, axis=1)  # Delete column
                del c[kton_column]
                lb = np.delete(lb, kton_column)
                ub = np.delete(ub, kton_column)
                del variable_names[kton_column]

                cons_senses[first_kton_index] = '<='

                # Remove the negative counterpart
                A_new = np.delete(A_new, index_rows_to_delete[-1], axis=0)  # Delete row
                A = csr_matrix(A_new)
                del b[index_rows_to_delete[-1]]
                del cons_senses[index_rows_to_delete[-1]]

                save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
                copied_model = build_model_from_json(current_matrices_path)
                copied_model.update()
                copied_model, track_elements = canonical_form(copied_model, minOption=True)
                copied_model.update()
            else:
                k -= 1
    copied_model.update()
    return copied_model, kton_dict


def eliminate_singleton_inequalities(model, current_matrices_path):
    """
    This function processes a Gurobi model to eliminate singleton inequality constraints.
    It simplifies the model by removing redundant constraints and updating the model accordingly.

    Args:
    model (gurobipy.Model): The Gurobi model to process.

    Returns:
    gurobipy.Model: The updated Gurobi model after eliminating singleton equalities.
    """

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Getting the matrices of the model
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

    # Getting the variable names
    variable_names = [var.VarName for var in copied_model.getVars()]

    # Getting the number of nonzero elements per row
    nonzero_count_per_row = np.count_nonzero(A.A, axis=1)

    # Create a boolean array where True indicates rows with a single non-zero element
    single_nonzero = nonzero_count_per_row == 1

    has_negative_counterpart, indices_list = find_corresponding_negative_rows_with_indices(A, b)
    negated_list = [not x for x in has_negative_counterpart]

    # Combine the two conditions
    valid_rows = np.logical_and(single_nonzero, negated_list)

    # Identify zero rows and classify constraints
    feedback_constraint = {i: 'Valid' for i in range(len(copied_model.getConstrs()))}
    feedback_variable = {v.VarName: 'valid' for v in copied_model.getVars()}

    to_delete_constraint = []
    to_delete_variable = []
    for i, constr in enumerate(copied_model.getConstrs()):
        if valid_rows[i]:

            # Extract the singleton row
            singleton_row = A.A[i, :]

            # Identify the index of the non-zero column (k)
            k_index = np.nonzero(singleton_row)[0][0]

            # Identify Aik
            A_ik = singleton_row[k_index]

            # Identify the right-hand-side
            b_i = b[i]

            if (A_ik > 0) and (b_i < 0):
                feedback_constraint[i] = 'Redundant'
                to_delete_constraint.append(i)
            elif (A_ik < 0) and (b_i > 0):
                feedback_constraint[i] = 'Infeasible'
            elif (A_ik > 0) and (b_i == 0):
                feedback_constraint[i] = 'Redundant'
            elif (A_ik < 0) and (b_i == 0):
                feedback_constraint[i] = 'Redundant'
                feedback_variable[copied_model.getVars()[k_index].VarName] = 'Redundant'
                to_delete_constraint.append(i)
                to_delete_variable.append(k_index)

    # Exclude constraints and variables
    A_new = np.delete(A.A, to_delete_constraint, axis=0)  # remove constraint (row)
    A_new = np.delete(A_new, to_delete_variable, axis=1)  # remove variable (column)
    A = csr_matrix(A_new)

    # Delete elements from b and the corresponding elements of the constraint operation
    # Iterating in reverse order to avoid index shifting issues
    for index in sorted(to_delete_constraint, reverse=True):
        del b[index]
        del cons_senses[index]

    for index in sorted(to_delete_variable, reverse=True):
        del c[index]
        del lb[index]
        del ub[index]
        del variable_names[index]

    save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
    model_copy = build_model_from_json(current_matrices_path)

    return model_copy, feedback_constraint, feedback_variable


def eliminate_dual_singleton_inequalities(model, current_matrices_path):
    """
    This function processes a Gurobi model to eliminate dual singleton inequality constraints.
    It simplifies the model by removing redundant variables and updating the model accordingly.

    Args:
    model (gurobipy.Model): The Gurobi model to process.

    Returns:
    gurobipy.Model: The updated Gurobi model after eliminating singleton equalities.
    """

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Getting the matrices of the model
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

    # Getting the variable names
    variable_names = [var.VarName for var in copied_model.getVars()]

    # Getting the number of nonzero elements per column
    nonzero_count_per_column = np.count_nonzero(A.A, axis=0)

    # Create a boolean array where True indicates rows with a single non-zero element
    valid_columns = nonzero_count_per_column == 1

    # Identify zero rows and classify constraints
    feedback_constraint = {i: 'Valid' for i in range(len(copied_model.getConstrs()))}
    feedback_variable = {v.VarName: 'valid' for v in copied_model.getVars()}

    to_delete_constraint = []
    to_delete_variable = []
    for j, var in enumerate(model.getVars()):
        if valid_columns[j]:

            # Extract the singleton row
            singleton_column = A.A[:, j]

            # Identify the index of the non-zero column (k)
            r_index = np.nonzero(singleton_column)[0][0]

            # Identify Aik
            A_ik = singleton_column[r_index]

            # Identify the cost
            c_j = c[j]

            if (A_ik > 0) and (c_j < 0):
                feedback_variable[copied_model.getVars()[j].VarName] = 'Infeasible'
            elif (A_ik < 0) and (c_j > 0):
                feedback_variable[copied_model.getVars()[j].VarName] = 'Redundant'
                to_delete_variable.append(j)
            elif (A_ik > 0) and (c_j == 0):
                feedback_variable[copied_model.getVars()[j].VarName] = 'Redundant'
                to_delete_variable.append(j)
                feedback_constraint[r_index] = 'Redundant'
                to_delete_constraint.append(r_index)
            elif (A_ik < 0) and (c_j == 0):
                feedback_variable[copied_model.getVars()[j].VarName] = 'Redundant'
                to_delete_variable.append(j)

    # Exclude constraints and variables
    A_new = np.delete(A.A, to_delete_constraint, axis=0)  # remove constraint (row)
    A_new = np.delete(A_new, to_delete_variable, axis=1)  # remove variable (column)
    A = csr_matrix(A_new)

    # Delete elements from b and the corresponding elements of the constraint operation
    # Iterating in reverse order to avoid index shifting issues
    for index in sorted(to_delete_constraint, reverse=True):
        del b[index]
        del cons_senses[index]

    for index in sorted(to_delete_variable, reverse=True):
        del c[index]
        del lb[index]
        del ub[index]
        del variable_names[index]

    save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
    model_copy = build_model_from_json(current_matrices_path)

    return model_copy, feedback_constraint, feedback_variable


def eliminate_redundant_columns(model, current_matrices_path):
    """
    Eliminates redundant columns and constraints from a linear programming model based on the provided conditions.

    Args:
        model: The linear programming model object.
        current_matrices_path: Path to the current matrices of the model for backup purposes.

    Returns:
        model_copy: A copy of the model with redundant columns and constraints removed.
        feedback_constraint: List of indices of constraints removed.
        feedback_variable: List of names of variables removed.
    """
    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Getting the matrices of the model
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

    # Getting the variable names
    variable_names = [var.VarName for var in copied_model.getVars()]

    # Create boolean arrays for each condition
    b_zero = np.array(b) == 0  # True for rows where b_i = 0

    # For each row, check if all non-zero elements are non-negative or all are non-positive
    non_negative_elements = np.all(A.A >= 0, axis=1)
    non_positive_elements = np.all(A.A <= 0, axis=1)

    # boolean for equalities
    has_negative_counterpart, indices_list = find_corresponding_negative_rows_with_indices(A, b)

    # Combined condition: True for rows where b_i = 0 and all elements are non-negative or all are non-positive
    combined_condition = b_zero & has_negative_counterpart & (non_negative_elements | non_positive_elements)

    # Identify zero rows and classify constraints
    feedback_constraint = {i: 'Valid' for i in range(len(copied_model.getConstrs()))}
    feedback_variable = {v.VarName: 'valid' for v in copied_model.getVars()}

    to_delete_constraint = []
    to_delete_variable = []

    for i, constr in enumerate(copied_model.getConstrs()):
        if combined_condition[i]:

            # finding the constraints to remove
            if i not in to_delete_constraint:
                to_delete_constraint.append(i)
                feedback_constraint[i] = 'Redundant'

            # finding nonnull variables on that constraint to remove
            non_zero_variables = np.nonzero(A.A[i, :])[0]
            for index in non_zero_variables:
                if index not in to_delete_variable:
                    to_delete_variable.append(index)
                    var_name = copied_model.getVars()[index].VarName
                    feedback_variable[var_name] = 'Redundant'

    # Exclude constraints and variables
    A_new = np.delete(A.A, to_delete_constraint, axis=0)  # remove constraint (row)
    A_new = np.delete(A_new, to_delete_variable, axis=1)  # remove variable (column)
    A = csr_matrix(A_new)

    # Delete elements from b and the corresponding elements of the constraint operation
    # Iterating in reverse order to avoid index shifting issues
    for index in sorted(to_delete_constraint, reverse=True):
        del b[index]
        del cons_senses[index]

    lb = np.delete(lb, to_delete_variable)
    ub = np.delete(ub, to_delete_variable)

    for index in sorted(to_delete_variable, reverse=True):
        del c[index]
        del variable_names[index]

    save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
    copied_model = build_model_from_json(current_matrices_path)

    return copied_model, feedback_constraint, feedback_variable


def eliminate_redundant_rows(model, current_matrices_path):
    """
    Eliminates redundant rows from a linear programming model's constraints matrix.

    This function copies the input model to avoid modifying the original, identifies redundant
    constraints using Gaussian elimination with partial pivoting, and updates the model by removing
    these constraints. It then saves the updated model matrices to a specified path and rebuilds the
    model from these matrices.

    Parameters:
    - model: The linear programming model to be processed.
    - current_matrices_path: Path to save the updated model matrices.

    Returns:
    - copied_model: The updated model with redundant rows removed.
    - feedback_constraint: A dictionary with feedback on each constraint (valid, redundant, infeasible).
    """

    # Copy the model to avoid modifying the original
    copied_model = model.copy()

    # Getting the matrices of the model
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(copied_model)

    dependent_rows, has_linear_dependency = linear_dependency(A)

    # boolean for equalities
    has_negative_counterpart, indices_list = find_corresponding_negative_rows_with_indices(A, b)

    # Initialize feedback dictionaries and deletion list
    feedback_constraint = {i: 'Valid' for i in range(len(cons_senses))}
    to_delete_constraint = []

    # Iterate over the constraints
    for i, constr in enumerate(copied_model.getConstrs()):
        if i not in to_delete_constraint:
            if has_linear_dependency[i]:
                list_of_dependencies = dependent_rows[i]
                negative_counterpart_index = indices_list[i]
                for j in list_of_dependencies:
                    if j != negative_counterpart_index:
                        if j not in to_delete_constraint:
                            feedback_constraint[i] = 'Redundant'
                            to_delete_constraint.append(j)

    # Exclude constraints
    A_new = np.delete(A.A, to_delete_constraint, axis=0)  # remove constraint (row)
    A = csr_matrix(A_new)

    # Delete elements from b and the corresponding elements of the constraint senses
    for index in sorted(to_delete_constraint, reverse=True):
        del b[index]
        del cons_senses[index]

    # Save updated matrices and rebuild the model
    save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
    copied_model = build_model_from_json(current_matrices_path)
    copied_model.update()

    return copied_model, feedback_constraint
