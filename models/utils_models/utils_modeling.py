import os
import json
import scipy.sparse as sp
import gurobipy as gp
import random
import numpy as np
from gurobipy import LinExpr
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_original_model(n_variables, n_constraints):
    """
    Create a random linear programming (LP) model with the specified number of decision variables and constraints.

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
        c = [random.uniform(-10, 10) for _ in range(n_variables)]
        model.setObjective(gp.quicksum(c[i] * variables[i] for i in range(n_variables)), sense=gp.GRB.MINIMIZE)

        # Add random constraints
        A = np.random.randint(1, 101, size=(n_constraints, n_variables))
        b = np.random.randint(1, 101, size=(n_constraints))
        for i in range(n_constraints):
            model.addConstr(gp.quicksum(A[i, j] * variables[j] for j in range(n_variables)) <= b[i],
                            name=f'Constraint_{i}')

        # Update model
        model.update()

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        model = None

    return model


def get_model_matrices(model):
    """
    Extract the constraint matrix (A), right-hand side vector (b), and objective function coefficients (c) from a Gurobi model.

    Parameters:
    - model (gurobipy.Model): The Gurobi model from which to extract matrices.

    Returns:
    - A (numpy.ndarray): Constraint matrix A.
    - b (numpy.ndarray): Right-hand side vector b.
    - c (numpy.ndarray): Objective function coefficients c.
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

    return A, b, c, lb, ub


def save_json(A, b, c, lb, ub, save_path):
    """
    Save matrices and data structures as JSON files.

    Parameters:
    - A (scipy.sparse.csr_matrix): Constraint matrix as a CSR matrix.
    - b (list): Right-hand side (RHS) vector as a list.
    - c (list): Objective function coefficients as a list.
    - lb (ndarray): Lower bounds as a NumPy ndarray.
    - ub (ndarray): Upper bounds as a NumPy ndarray.
    - save_path (str): Path to save JSON files.

    Example usage:
    save_json(A, b, c, lb, ub, 'data_path')

    Data types:
    - A: scipy.sparse.csr_matrix
    - b: list
    - c: list
    - lb: ndarray
    - ub: ndarray
    """

    # Create a dictionary to store the data
    data_dict = {
        'A': A.toarray(),  # Convert csr_matrix to dense array for JSON
        'b': b,
        'c': c,
        'lb': lb,
        'ub': ub
    }

    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Save each data item as a separate JSON file
    for name, data in data_dict.items():
        file_name = os.path.join(save_path, f'{name}.json')

        if isinstance(data, list):
            with open(file_name, 'w') as file:
                json.dump(data, file)
        elif isinstance(data, np.ndarray):
            data_list = data.tolist()
            with open(file_name, 'w') as file:
                json.dump(data_list, file)
        else:
            raise ValueError(f"Unsupported data type for {name}")


def build_model_from_json(data_path):
    """
    Build a Gurobi model from JSON files containing matrices and data.

    Parameters:
    - data_path (str): Path to the directory containing JSON files (A.json, b.json, c.json, lb.json, ub.json).

    Returns:
    - model (gurobipy.Model): Gurobi model constructed from the provided data.

    This function assumes that matrix A is in scipy.sparse.csr_matrix format, and vectors b, c, lb, and ub
    are in list or numpy.ndarray format. It creates a new Gurobi model, adds variables with bounds (lb and ub),
    sets the objective function using vector c, and adds constraints based on matrix A and vector b.

    Example usage:
    model = build_model_from_json('data_path')
    """

    # Define the file names for JSON files
    file_names = ['A.json', 'b.json', 'c.json', 'lb.json', 'ub.json']

    # Initialize data variables
    A = None
    b = None
    c = None
    lb = None
    ub = None

    # Load data from JSON files
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)

            if file_name == 'A.json':
                # Convert the loaded dense matrix back to a csr_matrix
                dense_matrix = np.array(data)
                A = sp.csr_matrix(dense_matrix)
            elif file_name == 'b.json':
                b = np.array(data)
            elif file_name == 'c.json':
                c = np.array(data)
            elif file_name == 'lb.json':
                lb = np.array(data)
            elif file_name == 'ub.json':
                ub = np.array(data)

    # Ensure A is a scipy.sparse.csr_matrix
    if not isinstance(A, sp.csr_matrix):
        raise ValueError("Matrix A is not in csr_matrix format")

    # Create a Gurobi model and add variables
    num_variables = len(c)
    model = gp.Model()
    x = model.addMVar(shape=num_variables, lb=lb, ub=ub, name='x')
    model.update()
    # Create the objective expression using quicksum
    objective_expr = gp.quicksum(c[i] * x[i] for i in range(num_variables))

    # Set the objective function to minimize
    model.setObjective(objective_expr, gp.GRB.MINIMIZE)

    # Add constraints using matrix A and vector b
    for i in range(A.shape[0]):
        start = A.indptr[i]
        end = A.indptr[i + 1]
        variables = A.indices[start:end]
        coefficients = A.data[start:end]

        constraint_expr: LinExpr = gp.quicksum(coefficients[j] * x[variables[j]] for j in range(len(variables)))
        model.addConstr(constraint_expr <= b[i], name=f'constraint_{i}')
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
        max_value = row.max()

        if max_value != 0:
            norm_A[i] = row / max_value
            scalers[i] = max_value

    return norm_A, scalers


def reduction_features(threshold, A_norm, A):
    """
    Reduce features of a matrix A based on a threshold applied to its normalized form A_norm.

    Parameters:
    threshold (float): The threshold value to apply.
    A_norm (scipy.sparse.csr_matrix): The normalized matrix.
    A (scipy.sparse.csr_matrix): The original matrix.

    Returns:
    red_A (scipy.sparse.csr_matrix): The reduced matrix.
    """
    # Ensure A is in csr format for efficient row-wise operations
    A = A.tocsr()
    A_norm = A_norm.tocsr()

    # Create a copy of A to form red_A
    red_A = A.copy()

    # Iterate over each element in A_norm and set corresponding element in red_A to 0 if it's below the threshold
    for i in range(A_norm.shape[0]):
        for j in range(A_norm.shape[1]):
            if A_norm[i, j] < threshold:
                red_A[i, j] = 0

    return red_A


def sensitivity_analysis(data_path, original_model, params):
    """
    Perform a sensitivity analysis on a Gurobi model by varying a threshold that affects certain matrix elements.

    This function first solves the original model and then iterates over a range of threshold values. For each threshold:
    1. It normalizes the matrix A from the original model.
    2. Reduces A by setting elements below the threshold to zero.
    3. Saves the reduced matrix and other model parameters.
    4. Rebuilds and solves a new model from these parameters.
    5. Records the threshold value, objective function value, and decision variables.

    Parameters:
    data_path (str): Path to save and load data for the model.
    original_model (gurobipy.Model): The original Gurobi model.
    params (dict): Dictionary with 'max_threshold', 'init_threshold', 'step_threshold' values for the analysis.

    Returns:
    tuple of lists: (eps, of, dv) where
      - eps is a list of threshold values used,
      - of is a list of objective function values for each threshold,
      - dv is a list of decision variable values for each threshold.
    """

    A, b, c, lb, ub = get_model_matrices(original_model)

    # Calculate normalized A
    A_norm, _ = normalize_features(A)

    # Initialize lists to store results
    eps = [0]  # Start with 0 threshold
    of = [original_model.objVal]  # Start with the objective value of the original model
    dv = [np.array([var.x for var in original_model.getVars()])]  # Start with decision variables of original model
    changed_indices = [None]  # List to store indices changed at each threshold

    # Iterate over threshold values
    threshold = params['init_threshold']
    while threshold <= params['max_threshold']:
        # Calculate reduced A
        A_red = reduction_features(threshold, A_norm, A)

        # Convert A and A_red to dense format for comparison
        A_dense = A.toarray()
        A_red_dense = A_red.toarray()

        # Record indices where A has been changed
        indices = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1]) if
                   A_dense[i, j] != 0 and A_red_dense[i, j] == 0]

        # Save the matrices
        save_json(A_red, b, c, lb, ub, data_path)

        # Create a new model from the saved matrices
        iterative_model = build_model_from_json(data_path)

        # Solve the new model
        iterative_model.setParam('OutputFlag', 0)  # Optionally suppress Gurobi output
        iterative_model.optimize()

        # Update lists with results
        # Check if the model found a solution
        if iterative_model.status == gp.GRB.OPTIMAL:
            # Model found a solution
            eps.append(threshold)
            of.append(iterative_model.objVal)
            changed_indices.append(indices)
            dv.append(np.array([var.x for var in iterative_model.getVars()]))
            print(
                f"Threshold {np.round(threshold,4)} has changed {len(indices)} items, "
                f"final objective function is: {np.round(iterative_model.objVal,6)}")
        else:
            # Model did not find a solution
            eps.append(threshold)
            of.append(np.nan)  # Append NaN for objective function
            changed_indices.append(np.nan)
            dv.append(np.full(len(original_model.getVars()), np.nan))
            print(f"Threshold {threshold} has no feasible solution")

        # Delete the model to free up resources
        del iterative_model

        # Increment threshold
        threshold += params['step_threshold']

    return eps, of, dv, changed_indices


def visual_sensitivity_analysis(eps, of, dv):
    # Filter non-NaN values and corresponding eps and dv
    filtered_eps = [eps[i] for i in range(len(of)) if not np.isnan(of[i])]
    filtered_of = [of[i] for i in range(len(of)) if not np.isnan(of[i])]
    filtered_dv = [dv[i] for i in range(len(of)) if not np.isnan(of[i])]

    # Plot for objective function
    fig_of = go.Figure()
    fig_of.add_trace(go.Scatter(x=filtered_eps, y=filtered_of, mode='lines+markers', name='Objective Function'))
    fig_of.update_layout(title='Objective Function Sensitivity Analysis', xaxis_title='Threshold',
                         yaxis_title='Objective Function Value')
    fig_of.show()

    # Determine number of basic decision variables (non-zero)
    num_variables = len(dv[0])
    num_basic_variables = sum([1 for var in dv[0] if var != 0])

    # Create subplots for decision variables
    fig_dv = make_subplots(rows=num_basic_variables, cols=1,
                           subplot_titles=[f'Decision Variable {i + 1}' for i in range(num_basic_variables)])
    row = 1
    for i in range(num_variables):
        if dv[0][i] != 0:  # Check if the variable is basic (non-zero in the first set of decision variables)
            values = [filtered_dv[j][i] for j in range(len(filtered_dv))]
            fig_dv.add_trace(go.Scatter(x=filtered_eps, y=values, mode='lines+markers', name=f'Variable {i + 1}'),
                             row=row, col=1)
            row += 1

    fig_dv.update_layout(height=300 * num_basic_variables, title='Decision Variables Sensitivity Analysis',
                         showlegend=False)
    fig_dv.update_xaxes(title_text='Threshold')
    fig_dv.update_yaxes(title_text='Value')
    fig_dv.show()


def create_dual_model(data_path):
    """
    Create and return the dual model of a given primal linear programming problem.

    This function constructs the dual of a linear programming problem based on the matrices A, b, and c.
    The primal problem is assumed to have the following form:

    Minimize    Z = C^T X
    subject to  A X <= b
                X >= 0

    The corresponding dual problem is formulated as:

    Maximize    W = b^T y
    subject to  A^T y <= C
                y >= 0

    Parameters:
    data_path (str): Path to the directory containing JSON files for matrices A, b, and c.
                     The matrices should be stored as 'A.json', 'b.json', and 'c.json'.

    Returns:
    gurobipy.Model: The constructed dual Gurobi model.

    The function reads the matrices A, b, and c from the specified data path, then uses these to set up
    and return the dual model. The dual model includes dual variables corresponding to the primal constraints,
    and constraints derived from the primal objective function coefficients.

    Example usage:
    dual_model = create_dual_model("path_to_data")
    """
    # Load the matrices A, b, c, lb, ub
    with open(os.path.join(data_path, 'A.json'), 'r') as file:
        A = np.array(json.load(file))
    with open(os.path.join(data_path, 'b.json'), 'r') as file:
        b = np.array(json.load(file))
    with open(os.path.join(data_path, 'c.json'), 'r') as file:
        c = np.array(json.load(file))
    # lb and ub are not needed for the dual problem as X >= 0

    # Create the dual model
    dual_model = gp.Model("dual_model")

    # Add dual variables for each primal constraint
    y = dual_model.addMVar(shape=A.shape[0], lb=0, name="y")

    # Set the dual objective function
    dual_model.setObjective(b @ y, gp.GRB.MAXIMIZE)

    # Add dual constraints based on primal objective function coefficients
    for i in range(A.shape[1]):
        coef = A[:, i]
        dual_model.addConstr(coef @ y <= c[i])

    dual_model.update()
    return dual_model


def build_dual_model_from_json(data_path):
    """
    Build the dual of a Gurobi model from JSON files containing matrices and data, specifically for a linear
    programming problem of the form:
    Minimize Z = C^T X
    subject to: A X <= b, X >= 0

    The dual problem is formulated as:
    Maximize W = b^T y
    subject to: A^T y <= C, y >= 0

    Parameters:
    - data_path (str): Path to the directory containing JSON files (A.json, b.json, c.json, lb.json, ub.json).

    Returns:
    - dual_model (gurobipy.Model): The dual Gurobi model constructed from the provided data.

    This function assumes that matrix A is in scipy.sparse.csr_matrix format, and vectors b, c, lb, and ub
    are in list or numpy.ndarray format. It creates a new Gurobi model for the dual, adds variables,
    sets the dual objective function using vector b, and adds constraints based on matrix A^T and vector c.

    Example usage:
    dual_model = build_dual_model_from_json('data_path')
    """

    # Define the file names for JSON files
    file_names = ['A.json', 'b.json', 'c.json', 'lb.json', 'ub.json']

    # Initialize data variables
    A = None
    b = None
    c = None
    lb = None
    ub = None

    # Load data from JSON files
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)

            if file_name == 'A.json':
                # Convert the loaded dense matrix back to a csr_matrix
                dense_matrix = np.array(data)
                dense_matrix_transpose = dense_matrix.transpose()
                A = sp.csr_matrix(dense_matrix_transpose)
            elif file_name == 'b.json':
                c = np.array(data)
            elif file_name == 'c.json':
                b = np.array(data)

    # Ensure A is a scipy.sparse.csr_matrix
    if not isinstance(A, sp.csr_matrix):
        raise ValueError("Matrix A is not in csr_matrix format")

    # Create a Gurobi model and add variables
    num_variables = len(c)
    lb = np.full(num_variables, 0)
    ub = np.full(num_variables, np.inf)

    model = gp.Model()
    y = model.addMVar(shape=num_variables, lb=lb, ub=ub, name='y')
    model.update()
    # Create the objective expression using quicksum
    objective_expr = gp.quicksum(c[i] * y[i] for i in range(num_variables))

    # Set the objective function to maximize
    model.setObjective(objective_expr, gp.GRB.MAXIMIZE)

    # Add constraints using matrix A and vector c
    for i in range(A.shape[0]):
        start = A.indptr[i]
        end = A.indptr[i + 1]
        variables = A.indices[start:end]
        coefficients = A.data[start:end]

        constraint_expr: LinExpr = gp.quicksum(coefficients[j] * y[variables[j]] for j in range(len(variables)))
        model.addConstr(constraint_expr >= b[i], name=f'constraint_{i}')
        model.update()
    return model