import os
import json
import scipy.sparse as sp
import gurobipy as gp
import random
import numpy as np
from gurobipy import LinExpr


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
        A = np.random.rand(n_constraints, n_variables)
        b = np.random.rand(n_constraints)
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
