import os
import json
import scipy.sparse as sp
import gurobipy as gp
import random
import numpy as np
import pandas as pd
from gurobipy import LinExpr
import plotly.graph_objects as go
from tabulate import tabulate
import time
import sys
from gurobipy import GRB
from scipy.sparse import issparse
from collections import defaultdict


def nested_dict():
    """
    Create a nested defaultdict.

    This function returns a nested defaultdict, which is a dictionary that allows an arbitrary number of levels
    of nesting. It simplifies the process of creating and working with nested dictionaries by automatically
    creating inner dictionaries as needed.

    Returns:
    - defaultdict: A nested defaultdict object that allows arbitrary levels of nesting.
    """
    return defaultdict(nested_dict)


def create_original_model(n_variables, n_constraints):
    """
    Create a random linear programming (LP) model with the specified number of decision variables and constraints.
    Minimize Z = C^T X
    subject to: A X >= b, X >= 0

    Parameters:
    - n_variables (int): Number of decision variables.
    - n_constraints (int): Number of constraints.

    Returns:
    - model (gurobipy.Model): The created LP model.
    """

    # Create a Gurobi model
    model = gp.Model("Original_LP")

    try:
        # Define decision variables
        variables = {}
        for i in range(n_variables):
            variables[i] = model.addVar(name=f'x_{i}')

        # Set objective function with random coefficients
        c = [random.uniform(1, 10) for _ in range(n_variables)]
        model.setObjective(gp.quicksum(c[i] * variables[i] for i in range(n_variables)), sense=gp.GRB.MINIMIZE)

        # Add random constraints
        A = np.random.randint(1, 101, size=(n_constraints, n_variables))
        b = np.random.randint(1, 101, size=(n_constraints))
        for i in range(n_constraints):
            model.addConstr(gp.quicksum(A[i, j] * variables[j] for j in range(n_variables)) >= b[i],
                            name=f'Constraint_{i}')

        # Update model
        model.update()

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        model = None

    return model


def get_model_matrices(model):
    """
    Extract key matrices and vectors from a Gurobi model. This function retrieves the constraint matrix (A),
    right-hand side vector (b), objective function coefficients (c), and the lower (lb) and upper bounds (ub) of the
    decision variables.

    Parameters:
    - model (gurobipy.Model): The Gurobi model from which to extract the matrices and vectors.

    Returns:
    - A (scipy.sparse.csr_matrix): The constraint matrix A in CSR (Compressed Sparse Row) format.
    - b (numpy.ndarray): The right-hand side vector b, containing the limits for each constraint.
    - c (numpy.ndarray): The objective function coefficients c, representing the coefficients of the decision variables
    in the objective function.
    - lb (numpy.ndarray): An array of lower bounds for each decision variable.
    - ub (numpy.ndarray): An array of upper bounds for each decision variable.
    - of_sense (int): The sense of the optimization (minimize or maximize).
    - cons_senses (list): A list of senses for each constraint in the model.
    """

    # Access constraint matrix A
    A = model.getA()

    # Access right-hand side (RHS) vector b
    b = model.getAttr('RHS', model.getConstrs())

    # Access objective function coefficients c
    c = model.getAttr('Obj', model.getVars())

    # Access lower bounds (lb) and upper bounds (ub)
    lb = np.array([var.LB for var in model.getVars()])
    ub = np.array([var.UB for var in model.getVars()])

    # Access the sense of optimization
    of_sense = model.ModelSense

    # get the constant in the of
    of = model.getObjective()
    co = of.getConstant()

    # Access the sense of each constraint
    cons_senses = [constr.Sense for constr in model.getConstrs()]

    # Getting the variable names
    variable_names = [var.VarName for var in model.getVars()]

    return A, b, c, co, lb, ub, of_sense, cons_senses, variable_names


def save_json(A, b, c, lb, ub, of_sense, cons_senses, save_path, co=0, variable_names=None):
    """
    Save matrices and data structures as JSON files, including the sense of optimization and constraints.

    Parameters:
    - A (scipy.sparse.csr_matrix): Constraint matrix as a CSR matrix.
    - b (numpy.ndarray): Right-hand side (RHS) vector.
    - c (numpy.ndarray): Objective function coefficients.
    - lb (numpy.ndarray): Lower bounds of variables.
    - ub (numpy.ndarray): Upper bounds of variables.
    - of_sense (int): Sense of optimization (1 for minimize, -1 for maximize).
    - cons_senses (list): List of senses for each constraint.
    - save_path (str): Path to save JSON files.
    - co (float, optional): constant of objective function. Default is 0
    - variable_names (list, optional): Names of the decision variables. If None, defaults to x1, x2, ...

    The data includes the constraint matrix A, vectors b, c, lower bounds lb, upper bounds ub,
    the sense of optimization, and the senses of each constraint.
    """

    # Generate default variable names if not provided
    if variable_names is None:
        variable_names = [f'x{i + 1}' for i in range(A.shape[1])]

    # Create a dictionary to store the data
    data_dict = {
        'A': A.toarray(),  # Convert csr_matrix to dense array for JSON
        'b': b,
        'c': c,
        'co': co,
        'lb': lb,
        'ub': ub,
        'of_sense': of_sense,
        'cons_senses': cons_senses,
        'variable_names': variable_names
    }

    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Save each data item as a separate JSON file
    for name, data in data_dict.items():
        file_name = os.path.join(save_path, f'{name}.json')

        if isinstance(data, list) or isinstance(data, np.ndarray):
            data_list = data if isinstance(data, list) else data.tolist()
            with open(file_name, 'w') as file:
                json.dump(data_list, file)
        else:
            with open(file_name, 'w') as file:
                json.dump(data, file)


def build_model_from_json(data_path):
    """
    Build a Gurobi model from JSON files containing matrices, data, and model specifications.

    Parameters:
    - data_path (str): Path to the directory containing JSON files (A.json, b.json, c.json, lb.json, ub.json,
    of_sense.json, cons_senses.json).

    Returns:
    - model (gurobipy.Model): Gurobi model constructed from the provided data.
    """

    # Define the file names for JSON files
    file_names = ['A.json', 'b.json', 'c.json', 'lb.json', 'ub.json', 'of_sense.json', 'cons_senses.json', 'co.json',
                  'variable_names.json']

    # Initialize data variables
    A, b, c, lb, ub, of_sense, cons_senses, co, variable_names = None, None, None, None, None, None, None, None, None

    # Load data from JSON files
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

                if file_name == 'A.json':
                    A = sp.csr_matrix(np.array(data))
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

    # Add variables with provided names or default to x1, x2, etc.
    x = []
    for i in range(num_variables):
        var_name = variable_names[i] if variable_names is not None and i < len(variable_names) else f'x{i + 1}'
        x.append(model.addVar(lb=lb[i], ub=ub[i], name=var_name))
    model.update()

    # Set objective function
    objective_expr = gp.quicksum(c[i] * x[i] for i in range(num_variables))
    if co is not None:
        objective_expr += co

    model.setObjective(objective_expr, of_sense)

    # Add constraints
    for i in range(A.shape[0]):
        start = A.indptr[i]
        end = A.indptr[i + 1]
        variables = A.indices[start:end]
        coefficients = A.data[start:end]

        constraint_expr = gp.quicksum(coefficients[j] * x[variables[j]] for j in range(len(variables)))
        if cons_senses[i] == '<' or cons_senses[i] == '<=':
            model.addConstr(constraint_expr <= b[i], name=f'constraint_{i}')
        elif cons_senses[i] == '>' or cons_senses[i] == '>=':
            model.addConstr(constraint_expr >= b[i], name=f'constraint_{i}')
        elif cons_senses[i] == '=':
            model.addConstr(constraint_expr == b[i], name=f'constraint_{i}')

    model.update()

    return model


def compare_models(model1, model2):
    """
    Compare two Gurobi models based on their objective values and decision variables.

    Parameters:
    model1, model2: Two Gurobi models to compare.

    Returns:
    - obj_deviation: Deviation between the objective values of the two models.
    - avg_var_deviation: Average variation between the decision variables of the models.
                          Returns NaN if the number of decision variables is not the same.
    """

    # 1. Calculate the deviation between the objective function values
    obj_deviation = abs(model1.objVal - model2.objVal)

    # 2. Check if they have the same number of decision variables
    vars1 = model1.getVars()
    vars2 = model2.getVars()

    if len(vars1) != len(vars2):
        avg_var_deviation = np.nan
    else:
        # Calculate the average variation between the decision variables values
        var_diffs = [abs(vars1[i].x - vars2[i].x) for i in range(len(vars1))]
        avg_var_deviation = sum(var_diffs) / len(vars1)

    return obj_deviation, avg_var_deviation


def normalize_features(A):
    """
    Normalize a CSR matrix by rows, but only for rows where the maximum value is not zero.

    Parameters:
    A (scipy.sparse.csr_matrix): The matrix to be normalized.

    Returns:
    norm_A (scipy.sparse.csr_matrix): The normalized matrix.
    scalers (numpy.array): Scale factors used for normalization (maximum value of each row, 1 for rows with max value of 0).
    """

    # Initialize an array to store the scale factors
    scalers = np.ones(A.shape[0])  # Default to 1 for rows where max value is zero

    # Create a copy of A to avoid modifying the original matrix
    norm_A = A.copy()

    # Normalize each row
    for i in range(A.shape[0]):
        row = A.getrow(i)
        max_value = np.abs(row).max()

        if max_value != 0:
            norm_A[i] = row / max_value
            scalers[i] = max_value

    return norm_A, scalers


def matrix_sparsification(threshold, A_norm, A):
    """
    Reduce features of a matrix A based on a threshold applied to its normalized form A_norm.
    Used to note how the coefficients affect the overall problem
    
    Parameters:
    threshold (float): The threshold value to apply.
    A_norm (scipy.sparse.csr_matrix): The normalized matrix.
    A (scipy.sparse.csr_matrix): The original matrix.

    Returns:
    red_A (scipy.sparse.csr_matrix): The reduced matrix.
    """
    # Convert A to lil format for efficient row-wise operations
    A = A.tolil()
    A_norm = A_norm.tolil()

    # Create a copy of A to form red_A
    red_A = A.copy()

    # Iterate over each element in A_norm and set corresponding element in red_A to 0 if it's below the threshold
    for i in range(A_norm.shape[0]):
        for j in range(A_norm.shape[1]):
            if abs(A_norm[i, j]) < threshold:
                red_A[i, j] = 0

    # Convert red_A back to csr format
    return red_A.tocsr()


def sparsification_sensitivity_analysis(sens_data, model, params, model_to_use='primal'):
    """
    Perform a sensitivity analysis on a Gurobi model by varying a threshold that affects certain matrix elements.

    This function first solves the original model and then iterates over a range of threshold values. For each threshold:
    1. It normalizes the matrix A from the original model.
    2. Reduces A by setting elements below the threshold to zero.
    3. Saves the reduced matrix and other model parameters.
    4. Rebuilds and solves a new model from these parameters.
    5. Records the threshold value, objective function value, and decision variables.

    Parameters:
    sens_data (str): Path to save and load data for the model.
    original_model (gurobipy.Model): The original Gurobi model.
    params (dict): Dictionary with 'max_threshold', 'init_threshold', 'step_threshold' values for the analysis.

    Returns:
    tuple of lists: (eps, of, dv) where
      - eps is a list of threshold values used,
      - of is a list of objective function values for each threshold,
      - dv is a list of decision variable values for each threshold.
    """

    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)

    # Calculate normalized A
    A_norm, _ = normalize_features(A)

    # Initialize lists to store results
    eps = [0]  # Start with 0 threshold
    dv = [np.array([var.x for var in model.getVars()])]  # Start with decision variables of original model
    changed_indices = [None]  # List to store indices changed at each threshold
    abs_vio, obj_val = measuring_constraint_infeasibility(model, dv[0])
    of = [obj_val]  # Start with the objective value of the original model
    constraint_viol = [abs_vio]  # List to store infeasibility results
    of_dec = [model.objVal]

    # Calculate total iterations
    total_iterations = int((params['max_threshold'] - params['init_threshold']) / params['step_threshold']) + 1
    start_time = time.time()
    iteration = 1  # Initialize iteration counter
    # Iterate over threshold values
    threshold = params['init_threshold']

    while threshold <= params['max_threshold']:
        # Calculate reduced A
        A_red = matrix_sparsification(threshold, A_norm, A)

        # Convert A and A_red to dense format for comparison
        A_dense = A.toarray()
        A_red_dense = A_red.toarray()

        # Record indices where A has been changed
        indices = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1]) if
                   A_dense[i, j] != 0 and A_red_dense[i, j] == 0]

        # Save the matrices
        save_json(A_red, b, c, lb, ub, of_sense, cons_senses, sens_data)

        # Create a new model from the saved matrices
        iterative_model = build_model_from_json(sens_data)

        # Solve the new model
        iterative_model.setParam('OutputFlag', 0)
        iterative_model.optimize()

        # Update lists with results
        # Check if the model found a solution
        if iterative_model.status == gp.GRB.OPTIMAL:
            # Model found a solution
            eps.append(threshold)
            of.append(iterative_model.objVal)
            changed_indices.append(indices)
            dv.append(np.array([var.x for var in iterative_model.getVars()]))
            # Access the decision variables
            variables = iterative_model.getVars()
            # Extract their values and store in a NumPy array
            decisions = np.array([var.x for var in variables])
            # calculate the constraint infeasibility
            abs_vio, obj_val = measuring_constraint_infeasibility(model, decisions)

            # Store infeasibility results
            of_dec.append(obj_val)
            constraint_viol.append(abs_vio)
            if params['prints']:
                print(
                    f"Threshold {np.round(threshold, 4)} has changed {len(indices)} items, "
                    f"in {model_to_use}, final objective function is: {np.round(iterative_model.objVal, 6)}")
        else:
            # Model did not find a solution
            eps.append(threshold)
            of.append(np.nan)  # Append NaN for objective function
            of_dec.append(np.nan)
            changed_indices.append(np.nan)
            constraint_viol.append(np.nan)
            dv.append(np.full(len(model.getVars()), np.nan))
            if params['prints']:
                print(f"Threshold {threshold} has no feasible solution")

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
        threshold += params['step_threshold']
    print("\n")
    return eps, of, dv, changed_indices, constraint_viol, of_dec


def visual_join_sensitivity(eps, of_primal, dv_primal, cviol_p, of_dual, dv_dual, cviol_d, title_n, primal_sense):
    """
    Visualize the sensitivity analysis of sparsification on both primal and dual linear programming models.

    Parameters:
    - eps (list): A list of sparsification thresholds used in the analysis.
    - of_primal (list): Objective function values for the primal model at each threshold.
    - dv_primal (list of lists): Decision variable values for the primal model at each threshold.
    - cviol_p (list of lists): Constraint violation values for the primal model at each threshold.
    - of_dual (list): Objective function values for the dual model at each threshold.
    - dv_dual (list of lists): Decision variable values for the dual model at each threshold.
    - cviol_d (list of lists): Constraint violation values for the dual model at each threshold.
    """

    # Getting the sense of optimization for primal and dual
    if primal_sense == 1:
        primal_sens = '(Min)'
        dual_sens = '(Max)'
    else:
        primal_sens = '(Max)'
        dual_sens = '(Min)'

    # Figure for Objective Function Sensitivity Analysis (Primal and Dual)
    fig_of = go.Figure()
    name_primal_1 = 'Primal Objective Function ' + primal_sens
    name_dual_1 = 'Dual Objective Function ' + dual_sens
    fig_of.add_trace(go.Scatter(x=eps, y=np.round(of_primal, 2), mode='lines+markers', name=name_primal_1))
    fig_of.add_trace(go.Scatter(x=eps, y=np.round(of_dual, 2), mode='lines+markers', name=name_dual_1))
    full_title_1 = 'Objective Function Sensitivity Analysis - ' + title_n
    fig_of.update_layout(title=full_title_1, xaxis_title='Threshold',
                         yaxis_title='Objective Function Value')
    fig_of.show()

    # Figure for Primal Constraint Violation
    fig_cviol_p = go.Figure()
    for i in range(len(cviol_p[0])):
        fig_cviol_p.add_trace(
            go.Scatter(x=eps, y=[cv[i] for cv in cviol_p], mode='lines+markers', name=f'Constraint {i + 1} (Primal)'))
    fig_cviol_p.update_layout(title='Primal Constraint Violation - ' + title_n, xaxis_title='Threshold',
                              yaxis_title='Violation')
    fig_cviol_p.show()

    # Figure for Dual Constraint Violation
    fig_cviol_d = go.Figure()
    for i in range(len(cviol_d[0])):
        fig_cviol_d.add_trace(
            go.Scatter(x=eps, y=[cv[i] for cv in cviol_d], mode='lines+markers', name=f'Constraint {i + 1} (Dual)'))
    fig_cviol_d.update_layout(title='Dual Constraint Violation - ' + title_n, xaxis_title='Threshold',
                              yaxis_title='Violation')
    fig_cviol_d.show()

    # Figure for Primal Decision Variables Sensitivity Analysis
    fig_dv_p = go.Figure()
    for i, dv in enumerate(zip(*dv_primal)):  # Transpose to iterate over each variable
        fig_dv_p.add_trace(go.Scatter(x=eps, y=dv, mode='lines+markers', name=f'Variable {i + 1}'))
    fig_dv_p.update_layout(title='Primal Decision Variables Sensitivity Analysis - ' + title_n, xaxis_title='Threshold',
                           yaxis_title='Decision Variable Value')
    fig_dv_p.show()

    # Figure for Dual Decision Variables Sensitivity Analysis
    fig_dv_d = go.Figure()
    for i, dv in enumerate(zip(*dv_dual)):  # Transpose to iterate over each variable
        fig_dv_d.add_trace(go.Scatter(x=eps, y=dv, mode='lines+markers', name=f'Variable {i + 1}'))
    fig_dv_d.update_layout(title='Dual Decision Variables Sensitivity Analysis - ' + title_n, xaxis_title='Threshold',
                           yaxis_title='Decision Variable Value')
    fig_dv_d.show()


def build_dual_model_from_json(data_path):
    """
    Build the dual of a Gurobi model from JSON files containing matrices, data, and model specifications.

    Parameters:
    - data_path (str): Path to the directory containing JSON files (A.json, b.json, c.json, lb.json, ub.json,
    of_sense.json, cons_senses.json).

    Important:
    - Dual is created based on the saved matrix data considering that minimization problems have only >= constraints
    and tha maximization problems have only <= constraints.

    Returns:
    - dual_model (gurobipy.Model): The dual Gurobi model constructed from the provided data.
    """

    # Define the file names for JSON files
    file_names = ['A.json', 'b.json', 'c.json', 'lb.json', 'ub.json', 'of_sense.json', 'cons_senses.json']

    # Initialize data variables
    A_t, rhs, cost, lb, ub, of_sense, cons_senses = None, None, None, None, None, None, None

    # Load data from JSON files
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)

            if file_name == 'A.json':
                A_t = sp.csr_matrix(np.array(data).transpose())
            elif file_name == 'b.json':
                cost = np.array(data)
            elif file_name == 'c.json':
                rhs = np.array(data)
            elif file_name == 'of_sense.json':
                of_sense = data
            elif file_name == 'cons_senses.json':
                cons_senses = data
            elif file_name == 'lb.json':
                primal_lb = np.array(data)
            elif file_name == 'ub.json':
                primal_ub = np.array(data)

    # Create a Gurobi model
    model = gp.Model()
    num_variables = len(cost)
    if of_sense == 1:  # Minimization
        lb = np.full(num_variables, 0)
        ub = np.full(num_variables, np.inf)
    else:  # Maximization
        lb = np.full(num_variables, -np.inf)
        ub = np.full(num_variables, 0)

    # Add variables with names starting from y1
    y = []
    for i in range(num_variables):
        y.append(model.addVar(lb=lb[i], ub=ub[i], name=f'y{i + 1}'))
    model.update()

    # Create the objective expression using quicksum
    objective_expr = gp.quicksum(cost[i] * y[i] for i in range(num_variables))

    # Set the objective function of the dual model
    model.setObjective(objective_expr, gp.GRB.MAXIMIZE if of_sense == 1 else gp.GRB.MINIMIZE)

    # Add constraints using matrix A and vector c
    for i in range(A_t.shape[0]):
        start = A_t.indptr[i]
        end = A_t.indptr[i + 1]
        variables = A_t.indices[start:end]
        coefficients = A_t.data[start:end]

        constraint_expr: LinExpr = gp.quicksum(coefficients[j] * y[variables[j]] for j in range(len(variables)))
        if primal_lb[i] == -np.inf and primal_ub[i] == +np.inf:  # asymmetrical
            model.addConstr(constraint_expr == rhs[i], name=f'constraint_{i}')
        elif primal_lb[i] == 0 and primal_ub[i] == +np.inf:  # symmetrical
            model.addConstr(constraint_expr <= rhs[i], name=f'constraint_{i}')
        elif primal_lb[i] == -np.inf and primal_ub[i] == 0:  # symmetrical
            model.addConstr(constraint_expr >= rhs[i], name=f'constraint_{i}')

        model.update()

    final_dual = pre_processing_model(model)

    return final_dual


def constraint_reduction(model, threshold, path):
    """
    Reduces the constraints of a given Gurobi model based on the Euclidean distance of each constraint's coefficients
    to the zero vector. Constraints with a distance less than the specified threshold are removed.

    Parameters:
    - model (gurobipy.Model): The original Gurobi LP model to be reduced.
    - threshold (float): The threshold for removing constraints based on their Euclidean distance to zero.
    - path (str): Path to save the modified model matrices.

    The function first extracts the model's matrices and normalizes the constraint matrix (A). It then calculates the
    Euclidean distance of each constraint to the zero vector. Constraints with a distance less than the threshold are
    identified as candidates for removal. The function then creates a new reduced model without these constraints and
    saves the modified matrices to the specified path.
    """

    # Extract matrices and vectors from the model
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)

    # Normalize A
    A_norm, _ = normalize_features(A)

    # Calculate Euclidean distance of each row in A to the zero vector
    distances = np.linalg.norm(A_norm.toarray(), axis=1)

    # Identify constraints to be removed based on threshold
    to_remove = [i for i, dist in enumerate(distances) if dist < threshold]
    A_reduced = np.delete(A.toarray(), to_remove, axis=0)
    b_reduced = np.delete(b, to_remove)

    # Save the matrices
    save_json(sp.csr_matrix(A_reduced), b_reduced, c, lb, ub, of_sense, cons_senses, path)

    # Create a new model from the saved matrices
    reduced_model = build_model_from_json(path)

    return reduced_model


def constraint_distance_reduction_sensitivity_analysis(sens_data, model, params, model_to_use='primal'):
    """
    Perform sensitivity analysis on a Gurobi model by varying a threshold that affects the number of constraints
    based on the Euclidean distance of each constraint to the zero vector.

    Parameters:
    sens_data (str): Path to save and load data for the model.
    model (gurobipy.Model): The original Gurobi model.
    params (dict): Dictionary with 'max_threshold', 'init_threshold', 'step_threshold' values for the analysis.

    Returns:
    tuple of lists: (eps, of, dv) where
      - eps is a list of threshold values used,
      - of is a list of objective function values for each threshold,
      - dv is a list of decision variable values for each threshold.
    """

    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)

    # Calculate normalized A
    A_norm, _ = normalize_features(A)

    # Initialize lists to store results
    eps = [0]  # Start with 0 threshold
    of = [model.objVal]  # Start with the objective value of the original model
    dv = [np.array([var.x for var in model.getVars()])]  # Start with decision variables of original model
    changed_constraints = [None]  # List to store constraints removed at each threshold
    constraint_viol = []  # List to store infeasibility results
    of_dec = [model.objVal]
    distances = np.linalg.norm(A_norm.toarray(), axis=1)  # Euclidean distance of each row in A to the zero vector

    # Iterate over threshold values
    threshold = params['init_threshold']
    while threshold <= params['max_threshold']:

        # Identify constraints to be removed based on threshold
        to_remove = [i for i, dist in enumerate(distances) if dist < threshold]
        A_reduced = np.delete(A.toarray(), to_remove, axis=0)
        b_reduced = np.delete(b, to_remove)

        # Save the matrices
        save_json(sp.csr_matrix(A_reduced), b_reduced, c, lb, ub, of_sense, cons_senses, sens_data)

        # Create a new model from the saved matrices
        iterative_model = build_model_from_json(sens_data)

        # Solve the new model
        iterative_model.setParam('OutputFlag', 0)  # suppress Gurobi output
        iterative_model.optimize()

        # Update lists with results
        # Check if the model found a solution
        if iterative_model.status == gp.GRB.OPTIMAL:
            # Model found a solution
            eps.append(threshold)
            of.append(iterative_model.objVal)
            changed_constraints.append(to_remove)
            dv.append(np.array([var.x for var in iterative_model.getVars()]))
            # Access the decision variables
            variables = iterative_model.getVars()
            # Extract their values and store in a NumPy array
            decisions = np.array([var.x for var in variables])
            # calculate the constraint infeasibility
            abs_vio, obj_val = measuring_constraint_infeasibility(model, decisions)
            # Store infeasibility results
            of_dec.append(obj_val)
            constraint_viol.append(abs_vio)
            print(
                f"Threshold {np.round(threshold, 4)} has removed {len(to_remove)} constraints, "
                f"in {model_to_use}, final objective function is: {np.round(iterative_model.objVal, 6)}")
        else:
            # Model did not find a solution
            eps.append(threshold)
            of.append(np.nan)  # Append NaN for objective function
            of_dec.append(np.nan)
            changed_constraints.append(to_remove)
            constraint_viol.append(np.full(len(model.getConstrs()), np.nan).tolist())
            dv.append(np.full(len(model.getVars()), np.nan))
            print(f"Threshold {threshold}: No feasible solution")

        # Delete the model to free up resources
        del iterative_model

        # Increment threshold
        threshold += params['step_threshold']

    return eps, of, dv, changed_constraints, constraint_viol, of_dec


def measuring_constraint_infeasibility(target_model, decisions):
    """
    Measure the infeasibility of a solution with respect to a given LP model's constraints.

    Parameters:
    - target_model (gurobipy.Model): The Gurobi model whose constraints are to be checked.
    - decisions (numpy.ndarray): Decision variable values from a modified model to be assessed.

    The function iterates over the constraints of the target model, calculating the difference between the LHS and RHS.
    It identifies violations based on the constraint type (<= or >=) and calculates the absolute and percentage violations.

    Returns:
    - A summary of violations, in absolute terms, for each constraint.
    """

    # Extract A, b and c from the target model
    A, b, c, _, _, _, _, _, _ = get_model_matrices(target_model)
    cost = np.array(c)

    # Initialize arrays to store violations
    abs_vio = []  # Absolute violations

    # Calculate the objective function value
    obj_val = np.dot(cost, decisions)

    # Iterate over the constraints
    for i in range(A.shape[0]):
        lhs = np.dot(A.getrow(i).toarray(), decisions)
        rhs = b[i]

        # Calculate absolute violation
        abs_violation = max(0, lhs - rhs) if target_model.getConstrs()[i].Sense == '<' else max(0, rhs - lhs)
        if isinstance(abs_violation, np.ndarray):
            abs_violation = np.sum(abs_violation)
        abs_vio.append(abs_violation)

    return abs_vio, obj_val


def pre_processing_model(model):
    """
    Pre-process a Gurobi model to ensure all constraints are in the standard format for
    linear programming (A X >= b for minimization problems and A X <= b for maximization problems).
    This function also handles equality constraints by splitting them into two inequalities.

    Parameters:
    - model (gurobipy.Model): The Gurobi model to be pre-processed.

    Returns:
    - pre_processed_model (gurobipy.Model): The pre-processed Gurobi model.
    """

    # Clone the original model to avoid altering it directly
    pre_processed_model = model.copy()

    # Find the decision variables
    existing_vars = pre_processed_model.getVars()

    # Handling variable bounds
    for var in existing_vars:
        lower_bound = var.LB
        upper_bound = var.UB

        if lower_bound > -gp.GRB.INFINITY and lower_bound != 0:
            # create a new constraint with this bound
            if model.ModelSense == 1:  # Minimization --> constraints are >=
                # Add a new constraint to the model
                pre_processed_model.addConstr(var >= lower_bound)
            else:  # Maximization --> constraints are <=
                pre_processed_model.addConstr(-var <= -lower_bound)

            # Set the variable to be free (unrestricted)
            var.setAttr(gp.GRB.Attr.LB, -gp.GRB.INFINITY)

        if upper_bound < gp.GRB.INFINITY and upper_bound != 0:
            # create a new constraint with this bound
            if model.ModelSense == 1:  # Minimization --> constraints are >=
                # Add a new constraint to the model
                pre_processed_model.addConstr(-var >= -upper_bound)
            else:  # Maximization --> constraints are <=
                pre_processed_model.addConstr(var <= upper_bound)

            # Set the variable to be free (unrestricted)
            var.setAttr(gp.GRB.Attr.UB, gp.GRB.INFINITY)

    # Update model after handling variable bounds
    pre_processed_model.update()

    # Iterate over each constraint in the model
    for constr in pre_processed_model.getConstrs():
        # Get the sense of the constraint
        sense = constr.Sense

        # For equality constraints, replace with two inequalities
        if sense == gp.GRB.EQUAL:
            # Get the linear expression and RHS value of the constraint
            lhs_expr = pre_processed_model.getRow(constr)
            rhs_value = constr.RHS

            # Add two new constraints to replace the equality
            pre_processed_model.addConstr(lhs_expr >= rhs_value, name=constr.ConstrName + "_geq")
            pre_processed_model.addConstr(lhs_expr <= rhs_value, name=constr.ConstrName + "_leq")

            # Remove the original equality constraint
            pre_processed_model.remove(constr)

    # Update model
    pre_processed_model.update()

    for constr in pre_processed_model.getConstrs():
        # Get the sense of the constraint
        sense = constr.Sense
        # For maximization problems, ensure all constraints are <=
        if model.ModelSense == -1 and sense == gp.GRB.GREATER_EQUAL:
            # Flip the constraint to <=
            lhs_expr = pre_processed_model.getRow(constr)
            rhs_value = constr.RHS
            pre_processed_model.addConstr(-1 * lhs_expr <= -1 * rhs_value, name=constr.ConstrName + "_leq")
            pre_processed_model.remove(constr)

        # For minimization problems, ensure all constraints are >=
        if model.ModelSense == 1 and sense == gp.GRB.LESS_EQUAL:
            # Flip the constraint to >=
            lhs_expr = pre_processed_model.getRow(constr)
            rhs_value = constr.RHS
            pre_processed_model.addConstr(-1 * lhs_expr >= -1 * rhs_value, name=constr.ConstrName + "_geq")
            pre_processed_model.remove(constr)

    return pre_processed_model


def print_model_in_mathematical_format(model):
    """
    Prints a Gurobi model in a mathematical format.

    Parameters:
    - model (gurobipy.Model): The Gurobi model to be printed.
    """

    # Print the objective function
    objective = 'Minimize\n' if model.ModelSense == 1 else 'Maximize\n'
    objective += str(model.getObjective()) + '\n'

    # Print the constraints
    constraints = 'Subject To\n'
    for constr in model.getConstrs():
        sense = constr.Sense
        if sense == '<':
            sense = '≤'
        elif sense == '>':
            sense = '≥'
        constraints += str(model.getRow(constr)) + ' ' + sense + ' ' + str(constr.RHS) + '\n'

    # Print variable bounds (if they are not default 0 and infinity)
    bounds = 'Bounds\n'
    for var in model.getVars():
        lb = '-inf' if var.LB <= -gp.GRB.INFINITY else str(var.LB)
        ub = '+inf' if var.UB >= gp.GRB.INFINITY else str(var.UB)
        bounds += f'{lb} <= {var.VarName} <= {ub}\n'

    print(objective)
    print(constraints)
    print(bounds)


def quality_check(original_primal_bp, original_primal, created_primal, created_dual, tolerance=1e-6):
    """
    Performs a quality check on the provided optimization models, including a canonical model.

    Parameters:
    - original_primal_bp: The primal model before preprocessing.
    - original_primal: The primal model after preprocessing.
    - created_primal: The primal model created from matrices.
    - created_dual: The dual model created from matrices.
    - canonical_primal: The canonical form of the primal model.
    - track_elements: A dictionary tracking the changes made to variables and constraints in the canonical model.
    - tolerance: A float representing the tolerance for comparison (default is 1e-6).

    Raises:
    - ValueError: If any of the quality checks fail.

    Returns:
    - None
    """
    # Compare objective values of the primal models including the canonical model
    primal_models = [original_primal_bp, original_primal, created_primal]
    primal_models_names = ['original_primal_bp', 'original_primal', 'created_primal', 'canonical_primal']

    for model1 in primal_models:
        for model2 in primal_models:
            if abs(model1.objVal - model2.objVal) > tolerance:
                raise ValueError(
                    f"Quality check failed: Objective function values differ between {model1.ModelName} and {model2.ModelName}")

    # Building the dictionary of decision variables
    decision_vars = {}
    for ind, model in enumerate(primal_models):
        decision_vars[primal_models_names[ind]] = {}
        for var in model.getVars():
            if var.VarName.endswith("_pos"):
                original_var_name = var.VarName[:-4]  # Removing '_pos' to get the original variable name
                var_neg = model.getVarByName(f"{original_var_name}_neg")
                decision_vars[primal_models_names[ind]][original_var_name] = var.x - var_neg.x
            elif not var.VarName.endswith("_neg"):
                decision_vars[primal_models_names[ind]][var.VarName] = var.x

    # Comparing decision variable values across models
    primal_created_models = [original_primal, created_primal]
    primal_created_models_names = ['original_primal', 'created_primal', 'canonical_primal']

    for i in range(len(primal_created_models)):
        for j in range(i + 1, len(primal_created_models)):
            model1_name = primal_created_models_names[i]
            model2_name = primal_created_models_names[j]
            for var_name in decision_vars[model1_name]:
                if var_name in decision_vars[model2_name]:
                    var1_value = decision_vars[model1_name][var_name]
                    var2_value = decision_vars[model2_name][var_name]
                    if abs(var1_value - var2_value) > tolerance:
                        raise ValueError(
                            f"Quality check failed: Decision variable {var_name} differs between {model1_name} and {model2_name}")

    # Compare objective value of created_dual and original_primal_bp
    if abs(created_dual.objVal - original_primal_bp.objVal) > tolerance:
        raise ValueError(
            "Quality check failed: Objective function value differs between created_dual and original_primal_bp")


def sparsification_test(original_model, config, current_matrices_path):
    """
    Tests the effects of sparsification on a linear programming model.

    Parameters:
    original_model: The original Gurobi model.
    config: Configuration dictionary containing parameters for sparsification.
    current_matrices_path: Path to save the JSON representation of the sparse model.

    Returns:
    None: The function prints the results.
    """
    # Extract matrices and vectors from the original model
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_model)
    A_norm, A_scaler = normalize_features(A)

    # Apply matrix sparsification
    A_red = matrix_sparsification(config['test_sparsification']['threshold'], A_norm, A)

    # Save the sparse model to a JSON file
    save_json(A_red, b, c, lb, ub, of_sense, cons_senses, current_matrices_path)

    # Build and optimize the sparse model
    created_primal_red = build_model_from_json(current_matrices_path)
    created_primal_red.setParam('OutputFlag', config['verbose'])
    created_primal_red.optimize()

    # Extract decision variable values
    decisions = np.array([var.x for var in created_primal_red.getVars()])

    # Calculate the constraint infeasibility
    abs_vio, obj_val = measuring_constraint_infeasibility(original_model, decisions)

    # Print the results
    if created_primal_red.status == 2:
        print("============ Reduced model - Sparsification ============")
        print(f"Threshold: {config['test_sparsification']['threshold']}")
        print("Optimal Objective Value =", created_primal_red.objVal)
        print("Basic Decision variables: ")
        for var in created_primal_red.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

        print("\nConstraint Violations:")
        for idx, vio in enumerate(abs_vio):
            print(f"Constraint {idx + 1}: Absolute Violation = {vio}")

        total_violation = sum(abs_vio)
        print(f"Total Absolute Violation: {total_violation}")

    else:
        print("The sparsification results in an infeasible solution")


def constraint_reduction_test(original_model, config, current_matrices_path):
    """
    Tests the effects of constraint reduction on a linear programming model.

    Parameters:
    original_model: The original Gurobi model.
    config: Configuration dictionary containing parameters for constraint reduction.
    current_matrices_path: Path to save the reduced model.

    Returns:
    None: The function prints the results.
    """
    # Apply constraint reduction
    red_model = constraint_reduction(original_model, config['test_constraint_red']['threshold'], current_matrices_path)

    # Set verbosity and optimize the reduced model
    red_model.setParam('OutputFlag', config['verbose'])
    red_model.optimize()

    # Extract decision variable values
    decisions = np.array([var.x for var in red_model.getVars()])

    # Calculate the constraint infeasibility
    abs_vio, obj_val = measuring_constraint_infeasibility(red_model, decisions)

    # Print the results
    if red_model.status == 2:
        print("============ Reduced model - Constraints reduction ============")
        print(f"Threshold: {config['test_constraint_red']['threshold']}")
        print("Optimal Objective Value =", red_model.objVal)
        print("Basic Decision variables: ")
        for var in red_model.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

        print("\nConstraint Violations:")
        for idx, vio in enumerate(abs_vio):
            print(f"Constraint {idx + 1}: Absolute Violation = {vio}")

        total_violation = sum(abs_vio)
        print(f"Total Absolute Violation: {total_violation}")
    else:
        print("The reduction of constraints results in an infeasible solution")


def get_info_GAMS(directory, save_excel=False):
    """
    Reads GAMS optimization models in a directory and prints a summary table using tabulate.

    Parameters:
    directory: The directory containing GAMS models.
    save_excel: Boolean, if True, saves the summary table as an Excel file.

    Returns:
    None: Prints or saves the summary table.
    """
    data = []

    for filename in os.listdir(directory):
        if filename.endswith('.gms') or filename.endswith('.mps'):  # Add more file extensions if needed
            model_path = os.path.join(directory, filename)

            try:
                # Read the model
                model = gp.read(model_path)
                model.setParam('OutputFlag', 0)
                model.optimize()

                # Extract information
                num_constraints = model.NumConstrs
                num_variables = model.NumVars
                optimal_value = model.objVal if model.status == gp.GRB.OPTIMAL else np.nan

                # Append to the data list
                data.append([filename, num_constraints, num_variables, optimal_value])
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Create a DataFrame
    df = pd.DataFrame(data,
                      columns=['Name of LP', 'Number of Constraints', 'Number of Variables', 'Optimal Solution Value'])

    # Print the table using tabulate
    print(tabulate(df, headers='keys', tablefmt='grid'))

    if save_excel:
        output_file = os.path.join(directory, 'GAMS_Models_Summary.xlsx')
        df.to_excel(output_file, index=False)
        print(f"Summary saved to {output_file}")


def detailed_info_models(original_primal_bp, original_primal, created_primal, created_dual):
    """
    Prints detailed information and comparisons for a set of optimization models.

    Parameters:
    - original_primal_bp: The original primal model before pre-processing.
    - original_primal: The original primal model after pre-processing.
    - created_primal: The primal model created from matrices.
    - created_dual: The dual model created from matrices.
    """
    obj_var, dec_var = compare_models(original_primal, created_primal)
    print(f"Create model X original model: The absolute deviation of objective function value is {obj_var} "
          f"and the average deviation of decision variable values is {dec_var}.")

    if original_primal_bp.status == 2:
        print("============ Original Model (before pre-processing) ============")
        print("Optimal Objective Value =", original_primal_bp.objVal)
        print("Basic Decision variables: ")
        for var in original_primal_bp.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

    if original_primal.status == 2:
        print("============ Original Model ============")
        print("Optimal Objective Value =", original_primal.objVal)
        print("Basic Decision variables: ")
        for var in original_primal.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

    if created_primal.status == 2:
        print("============ Created Model ============")
        print("Optimal Objective Value =", created_primal.objVal)
        print("Basic Decision variables: ")
        for var in created_primal.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

    if created_dual.status == 2:
        print("============ Created Dual ============")
        print("Optimal Objective Value =", created_dual.objVal)
        print("Basic Decision variables: ")
        for var in created_dual.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)


def rhs_sensitivity(model):
    """
    Perform Right-Hand Side (RHS) sensitivity analysis on a Gurobi model.

    This function takes a Gurobi model as input, which is assumed to have been solved to optimality.
    It returns two vectors: 'allowable_decrease' and 'allowable_increase', which represent the amount by which
    the RHS of each constraint can be decreased or increased without altering the optimal basis of the solution.

    Args:
    model (gurobipy.Model): The Gurobi model for which RHS sensitivity analysis is to be performed.

    Returns:
    Tuple[List[float], List[float]]: Two lists containing the allowable decreases and increases in the RHS of each constraint.
    """

    allowable_decrease = []
    allowable_increase = []

    for constraint in model.getConstrs():
        # Append the allowable decrease and increase for each constraint
        allowable_decrease.append(constraint.SARHSLow)
        allowable_increase.append(constraint.SARHSUp)

    return allowable_decrease, allowable_increase


def cost_function_sensitivity(model):
    """
    Perform cost coefficient sensitivity analysis on a linear programming model.

    Args:
    model: The linear programming model, already solved to optimality.

    Returns:
    A dictionary containing the allowable increase and decrease for each variable's cost coefficient.
    """

    allowable_decrease = []
    allowable_increase = []

    # Iterate over all decision variables in the model
    for var in model.getVars():
        reduced_cost = var.RC

        if model.ModelSense == 1:
            # For minimization problems
            allowable_decrease.append(float('inf') if reduced_cost <= 0 else reduced_cost)
            allowable_increase.append(float('inf') if reduced_cost >= 0 else -reduced_cost)
        else:
            # For maximization problems
            allowable_decrease.append(float('inf') if reduced_cost >= 0 else -reduced_cost)
            allowable_increase.append(float('inf') if reduced_cost <= 0 else reduced_cost)

    return allowable_decrease, allowable_increase


def dict2json(dictionary, file_path):
    """
    Converts a dictionary to a JSON file, handling NumPy ndarray objects.
    Saves the JSON to the specified file path and logs the action.

    Parameters:
        dictionary (dict): The dictionary to be converted and saved.
        file_path (str): The path to save the JSON file.
    """

    # Function to convert ndarrays to lists
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

    # Convert and save as JSON
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, default=convert_ndarray, indent=4)


def get_constraint_expression(model, i):
    """
    Retrieves the i-th constraint from the Gurobi model and returns it as a string expression.

    Parameters:
    model (gurobipy.Model): The Gurobi model from which to extract the constraint.
    i (int): The index of the constraint to be retrieved.

    Returns:
    str: The constraint expressed in the form 'LHS operation RHS'.
         For example, '2x + 3y <= 5'.

    Raises:
    IndexError: If the index i is out of the range of the model's constraints.
    """
    # Ensure the index is within the range of constraints
    if i < 0 or i >= model.NumConstrs:
        raise IndexError("Constraint index is out of range.")

    # Retrieve the i-th constraint
    i_th_constraint = model.getConstrs()[i]

    # Get the linear expression (LHS) of the constraint
    lhs_expression = model.getRow(i_th_constraint)

    # Get the sense (equality/inequality) of the constraint
    constraint_sense = i_th_constraint.Sense

    # Get the RHS of the constraint
    rhs_value = i_th_constraint.RHS

    # Convert the sense symbol to a readable format
    sense_symbol = {
        GRB.LESS_EQUAL: "<=",
        GRB.GREATER_EQUAL: ">=",
        GRB.EQUAL: "="
    }.get(constraint_sense, "?")

    # Construct and return the full constraint expression
    return f"{lhs_expression} {sense_symbol} {rhs_value}"


def canonical_form(model, minOption=False):
    """
    Converts a given Gurobi model into its canonical form.

    In the canonical form, all variables are non-negative, and constraints are formatted uniformly as
    either <= or >= depending on the problem being a maximization or minimization problem. This
    function also handles bounded variables and equality constraints by converting them into the
    appropriate inequality form.

    Args:
    - model: The Gurobi model to be converted.
    - minOption: If True, ensures the optimization problem is set to minimization.

    Returns:
    - canonical_model: The converted model in canonical form.
    - track_elements: A dictionary tracking the changes made to variables and constraints
      (original or created).
    """

    # Initialize tracking dictionaries for variables and constraints
    track_elements = {'variables': {}, 'constraints': {}}

    # Clone the model to avoid changing the original
    canonical_model = model.copy()

    # Set the model sense to minimization if minOption is True
    if minOption and canonical_model.ModelSense != 1:
        # Multiply the objective function by -1 to switch from maximization to minimization
        canonical_model.setObjective(-1 * canonical_model.getObjective(), gp.GRB.MINIMIZE)

    # Handle variable bounds and convert them into constraints
    for var in canonical_model.getVars():
        lower_bound = var.LB
        upper_bound = var.UB

        # Handling lower bounds
        if lower_bound > -gp.GRB.INFINITY and lower_bound != 0:
            constr_name = f"{var.VarName}_lb"
            if canonical_model.ModelSense == 1:  # Minimization
                canonical_model.addConstr(var >= lower_bound, name=constr_name)
            else:  # Maximization
                canonical_model.addConstr(-var <= -lower_bound, name=constr_name)
            var.LB = -gp.GRB.INFINITY
            track_elements['constraints'][constr_name] = 'created'

        # Handling upper bounds
        if upper_bound < gp.GRB.INFINITY and upper_bound != 0:
            constr_name = f"{var.VarName}_ub"
            if canonical_model.ModelSense == 1:  # Minimization
                canonical_model.addConstr(-var >= -upper_bound, name=constr_name)
            else:  # Maximization
                canonical_model.addConstr(var <= upper_bound, name=constr_name)
            var.UB = gp.GRB.INFINITY
            track_elements['constraints'][constr_name] = 'created'

        track_elements['variables'][var.VarName] = 'original'

    canonical_model.update()

    # Handle equality constraints by converting them into two inequalities
    for constr in canonical_model.getConstrs():
        if constr.Sense == gp.GRB.EQUAL:
            lhs_expr = canonical_model.getRow(constr)
            rhs_value = constr.RHS
            canonical_model.addConstr(lhs_expr >= rhs_value, name=constr.ConstrName + "_geq")
            canonical_model.addConstr(lhs_expr <= rhs_value, name=constr.ConstrName + "_leq")
            track_elements['constraints'][constr.ConstrName + "_geq"] = 'created'
            track_elements['constraints'][constr.ConstrName + "_leq"] = 'created'
            canonical_model.remove(constr)
            track_elements['constraints'][constr.ConstrName] = 'removed'

    canonical_model.update()

    # Uniformize constraints to <= for maximization and >= for minimization
    for constr in list(canonical_model.getConstrs()):  # Convert to list to avoid modification during iteration
        lhs_expr = canonical_model.getRow(constr)
        rhs_value = constr.RHS

        if canonical_model.ModelSense == -1 and constr.Sense == gp.GRB.GREATER_EQUAL:  # Maximization
            canonical_model.addConstr(-lhs_expr <= -rhs_value, name=constr.ConstrName + "_leq")
            canonical_model.remove(constr)
            track_elements['constraints'][constr.ConstrName + "_leq"] = 'created'

        elif canonical_model.ModelSense == 1 and constr.Sense == gp.GRB.LESS_EQUAL:  # Minimization
            canonical_model.addConstr(-lhs_expr >= -rhs_value, name=constr.ConstrName + "_geq")
            canonical_model.remove(constr)
            track_elements['constraints'][constr.ConstrName + "_geq"] = 'created'

    canonical_model.update()

    # Transform variables with negative lower bounds to non-negative
    for var in list(canonical_model.getVars()):
        if var.LB < 0:
            # Create two auxiliary non-negative variables
            x_pos = canonical_model.addVar(lb=0, name=f"{var.VarName}_pos")
            x_neg = canonical_model.addVar(lb=0, name=f"{var.VarName}_neg")
            canonical_model.update()

            # Check if the variable is in the objective function and update it
            of = canonical_model.getObjective()
            variables_of = [of.getVar(i) for i in range(of.size())]
            for ind_o in range(len(variables_of)):
                if var.VarName == variables_of[ind_o].VarName:
                    obj_coeff = canonical_model.getObjective().getCoeff(0)
                    new_obj_expr = canonical_model.getObjective() + obj_coeff * (x_pos - x_neg - var)
                canonical_model.setObjective(new_obj_expr)

            # Replace original variable in all constraints
            for constr in list(canonical_model.getConstrs()):
                lhs_expr = canonical_model.getRow(constr)
                variables = [lhs_expr.getVar(i) for i in range(lhs_expr.size())]

                for ind in range(len(variables)):
                    if var.VarName == variables[ind].VarName:
                        var_coeff = lhs_expr.getCoeff(ind)
                        # Create a new linear expression
                        new_expr = gp.LinExpr(lhs_expr)
                        new_expr.addTerms([var_coeff, -var_coeff], [x_pos, x_neg])

                        # Update the constraint with the new expression
                        new_constr_name = f"{constr.ConstrName}_mod"
                        if constr.Sense == gp.GRB.LESS_EQUAL:
                            canonical_model.addConstr(new_expr <= constr.RHS, name=new_constr_name)
                        elif constr.Sense == gp.GRB.GREATER_EQUAL:
                            canonical_model.addConstr(new_expr >= constr.RHS, name=new_constr_name)
                        elif constr.Sense == gp.GRB.EQUAL:
                            canonical_model.addConstr(new_expr == constr.RHS, name=new_constr_name)

                        track_elements['constraints'][new_constr_name] = 'created'
                        canonical_model.remove(constr)
                        track_elements['constraints'][constr.ConstrName] = 'removed'

            canonical_model.remove(var)
            track_elements['variables'][var.VarName] = 'removed'
            track_elements['variables'][f"{var.VarName}_pos"] = {'type': 'created', 'original_var': var.VarName}
            track_elements['variables'][f"{var.VarName}_neg"] = {'type': 'created', 'original_var': var.VarName}

    canonical_model.update()

    return canonical_model, track_elements


def find_corresponding_negative_rows_with_indices(A, b):
    """
    Identify rows in matrix A that have a corresponding negative row and return the indices of such rows.

    Args:
    A (csr_matrix): The constraint matrix of the linear problem.
    b (list): The right-hand side values of the constraints.

    Returns:
    (np.array, list):
        - A boolean array indicating rows that have a corresponding negative row.
        - A list of lists with indices of rows that have negative counterparts, or None if there is no counterpart.
    """
    rows, cols = A.A.shape
    has_negative_counterpart = np.zeros(rows, dtype=bool)
    indices_list = [None] * rows  # Initialize with None

    for i in range(rows):
        if has_negative_counterpart[i]:  # Skip if already matched
            continue
        for j in range(i + 1, rows):  # Avoid comparing the row with itself and repeat comparisons
            # Check if A[j, :] is the negative of A[i, :] and b[j] is the negative of b[i]
            if np.allclose(A.A[i, :], -A.A[j, :]) and np.isclose(b[i], -b[j]):
                has_negative_counterpart[i] = True
                has_negative_counterpart[j] = True
                # Store indices of rows that are negatives of each other
                indices_list[i] = j
                indices_list[j] = i

    # Replace unmodified None with empty lists or keep it as per requirement
    indices_list = [indices if indices is not None else None for indices in indices_list]

    return has_negative_counterpart, indices_list


def linear_dependency(A, feasibility_tolerance=0.01):
    """
    This function checks for linear dependency among the rows of matrix A.

    Linear dependency is checked through division of corresponding row elements.
    If the division of any corresponding row elements across two rows has very little variation,
    it indicates a row is linearly dependent on the other.

    Parameters:
    - A: numpy.ndarray or scipy sparse matrix, matrix of coefficients of the linear constraints.
    - feasibility_tolerance: float, the tolerance limit under which two constrained row values
                             are considered the same (indicating low variety, hence, dependence).

    Returns:
    - vector_index: list of tuple, the recorded pairs of indices of rows that were compared
                    and found to have equal implications of each other's space.
    - any_dependency: bool, True if there is at least one set of rows in A that are
                      linearly dependent within the threshold; False otherwise.
    """

    # Convert sparse matrix to dense if necessary.
    if issparse(A):
        A = A.toarray()

    m, n = A.shape
    dependent_rows = [[] for _ in range(m)]  # Initialize with empty lists
    has_linear_dependency = np.zeros(m, dtype=bool)

    for i in range(m):
        for j in range(i + 1, m):
            with np.errstate(divide='ignore', invalid='ignore'):
                # Perform row element division, safety check for division by 0
                is_nonzero = (A[j, :] != 0)
                div = np.where(is_nonzero, A[i, :] / A[j, :], np.inf)
                div_filtered = div[np.isfinite(div)]

                if len(div_filtered) > 0:
                    # Choose an element with a real value for comparison
                    div_with_value = div_filtered[0]
                    # Check if the differences are within the tolerance
                    close_enough = np.all(np.abs(div_filtered - div_with_value) < feasibility_tolerance)
                    if close_enough:
                        has_linear_dependency[i] = True
                        dependent_rows[i].append(j)
                        dependent_rows[j].append(i)

    return dependent_rows, has_linear_dependency


def model_stats(model):
    """
    Calculates the number of variables and constraints in a Gurobi model.

    Parameters:
    - model: Gurobi Model object, the optimization model from which to calculate the stats.

    Returns:
    - n_var: int, the number of variables in the model.
    - n_const: int, the number of constraints in the model.
    """
    n_var = model.numVars
    n_const = model.numConstrs
    return n_var, n_const
