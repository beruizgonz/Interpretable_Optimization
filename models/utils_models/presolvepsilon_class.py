import numpy as np
from scipy.sparse import csr_matrix, spdiags, diags
from utils_models.utils_functions import get_model_matrices, \
    find_corresponding_negative_rows_with_indices, nested_dict, normalize_features_sparse, normalize_features, \
    matrix_sparsification, calculate_bounds, build_dual_model_from_json, build_model_from_json, save_json, build_model
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime
import logging
import os
import gurobipy as gp   
from gurobipy import GRB
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.linalg import qr

import numpy as np
from scipy.sparse import issparse, csc_matrix

def flexibility_constraints(model, constraint_names, vars_remove):
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
    print(len(vars_remove))
    new_vars = []
    for row in vars_remove:
        constraint_name = constraint_names[row]  # Get the constraint name

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

    model.update()  # Ensure variables are added before modifying constraints
    print(len(new_vars))
    # Modify constraints to include excess and slack variables
    for constr in model.getConstrs():
        constr_name = constr.ConstrName  # Get the constraint name
        rhs = constr.RHS  # Get the right-hand side value
        excess_var = model.getVarByName(f"excess_{constr_name}")
        slack_var = model.getVarByName(f"defect_{constr_name}")
        # Add the new constraint
        if excess_var and slack_var:
            row_expr = model.getRow(constr)  # Get the existing constraint expression
            
            # Modify the constraint directly
            row_expr += excess_var - slack_var  # Adjust the expression

            model.chgCoeff(constr, excess_var, 1)  # Add excess variable coefficient
            model.chgCoeff(constr, slack_var, -1) # Add the new constraint

    model.update()
    return model


def flexibility_constraints1(A, b, cons_senses, vars_remove):
    # obtain the min max of the variable for each row to reduce
    row_indices, col_indices = A.nonzero()
    # Dictionary to store min-max indices per row
    row_min_max = {}
    variables ={}
    for row, cols_to_remove in vars_remove.items():
        # Get the nonzero column indices and values for the row
        _, col_indices = A[row].nonzero()  # Get nonzero column indices in the row
        row_values = A[row].toarray().flatten()  # Get full row as a dense array
        # Filter only relevant columns and their values
        relevant_data = [(col, row_values[col]) for col in col_indices if col in cols_to_remove]
        variables[row] = [col for col in col_indices if col in cols_to_remove]
        if relevant_data:  # Ensure there is at least one valid column
            # Find min and max based on values
            min_col_index, min_value = min(relevant_data, key=lambda x: x[1])
            max_col_index, max_value = max(relevant_data, key=lambda x: x[1])

            row_min_max[row] = (min_value, max_value)
    #print(row_min_max)
    # Change the sense of the constraints
    new_vars = []
    print(list(row_min_max.items())[0:10])
    entro1 = 0
    entro2 = 0
    entro3 = 0
    for row, (min_col, max_col) in row_min_max.items():
        abs_min, abs_max = abs(min_col), abs(max_col)
        # Check the sign is the same
        if np.sign(min_col) == np.sign(max_col):
            if min_col < 0:
                entro1 += 1
                cons_senses[row] = '<='
            elif min_col > 0:
                entro2 += 1
                cons_senses[row] = '>='
        else:
            if abs_min < abs_max:
                entro3 += 1
                cons_senses[row] = '>='  # Change to equality
            else:
                entro3 += 1 
                cons_senses[row] = '<='  # Change to equality
            new_vars.append(variables[row])
    new_vars =set([item for sublist in new_vars for item in sublist])
    print('Number of variables to remove', len(new_vars))
    print(entro1, entro2, entro3)
    return cons_senses, new_vars 

class PresolvepsilonOperations:
    def __init__(self, model = None,
                 opts = None):

        """
        Initialize the class with the model and the parameters for the presolve operations.
        """

        self.model = model
        self.change_elements = True
        # Initialize placeholders for matrices and model components
        self.A = None
        self.b = None
        self.c = None
        self.co = None
        self.lb = None
        self.ub = None
        self.of_sense = None
        self.cons_senses = None
        self.variable_names = None
        self.constr_names = None
        self.original_row_index = None
        self.original_column_index = None

        self.opts = opts

        # Initialize dictionary to track changes made by presolve operations
        self.change_dictionary = defaultdict(nested_dict)

        # Initialize table to track the number of variables and constraints at each operation
        self.operation_table = []

    def load_model_matrices(self, model):
        """
        Extract and load matrices and other components from the optimization model.
        """

        if self.model is None:
            raise ValueError("Model is not provided.")

        if self.opts.bounds:
            print("Calculating bounds")
            lb, ub_new = calculate_bounds(model)
            path = os.path.join(os.getcwd(), 'model_matrices.json')
            model_bounds = build_model_from_json(path)
            model_bounds.setParam('OutputFlag', 0)
            model_bounds.optimize()
            print('Model optimized: ', model_bounds.objVal)
        else:
            print("Using the model matrices")
            model_bounds = model
        self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names, self.constr_names = (
            get_model_matrices(model_bounds))
        ub = np.array(self.ub)
        print('Finite upper: ', np.sum(np.isfinite(ub)), 'Total upper: ', len(ub))

    def orchestrator_presolve_operations(self, model, epsilon = 1e-6):

        self.load_model_matrices(model)
        if self.opts.operate_epsilon_rows:
            print("Eliminate zero rows operation")
            self.eliminate_zero_rows_operation(epsilon = epsilon, norm_abs = self.opts.norm_abs)

        if self.opts.operate_epsilon_cols:
            print("Eliminate zero columns operation")
            self.eliminate_zero_columns_operation(epsilon = epsilon, norm_abs = self.opts.norm_abs)

        if self.opts.sparsification:
            print("Sparsification operation")
            self.sparsification_operation(epsilon = epsilon)
            #self.improve_sparsity(epsilon = epsilon)

        if self.opts.dependency_rows:
            print("Linear dependency rows")
            self.find_basis(epsilon = epsilon)

        return (self.A, self.b, self.c, self.lb, self.ub, self.of_sense, self.cons_senses, self.co, self.variable_names,
                self.change_dictionary, self.operation_table)


    def sparsification_operation(self,epsilon = 1e-6):
        """
        Perform sparsification operation on the model matrices.
        """
        A_norm = self.A.copy()
        A_norm1, _, _ = normalize_features_sparse(A_norm, self.b)
        self.ub = np.array(self.ub)
        # Covert the matrix to a CSR format
        A_norm = csr_matrix(self.A)
        A_norm1 = A_norm1.toarray()
        # Convert A_norm to an array
        A_dense = self.A.toarray()
        A_norm = A_norm.toarray()
        A_modifiable = A_dense.copy()
        max_sparsification = np.where(A_norm1 < 0, A_norm1* self.ub, np.where(A_norm1 > 0, A_norm1 * self.lb, 0))
        ones = np.ones((A_dense.shape[1], A_dense.shape[1]))
        np.fill_diagonal(ones, 0)
        max_activity_variable = max_sparsification @ ones
        upper_bound_matrix = A_norm1* self.ub
        #upper_bound_matrix = np.where(A_norm1 < 0, A_norm1 * self.lb, np.where(A_norm1 > 0, A_norm1 * self.ub, 0)) # tiene sentido???
        epsilon_matrix = np.where(
                        (np.isfinite(max_activity_variable)) & (max_activity_variable != 0) & (np.isfinite(upper_bound_matrix)),
                        #np.abs(upper_bound_matrix) / np.abs(max_activity_variable),
                        np.abs(upper_bound_matrix / max_activity_variable),
                        np.inf
                    )

        indices = np.where(epsilon_matrix < epsilon)
        A_modifiable[indices] = 0
        zero_rows = np.where(np.all(A_modifiable == 0, axis=1))[0]
        print("Zero rows: ", len(zero_rows))
        self.A = csr_matrix(A_modifiable)

    def eliminate_zero_rows_operation(self, epsilon, norm_abs='euclidean'):
        """
        Eliminate rows in the matrix where the normalized row norm relative to the RHS vector
        is below a given epsilon threshold.

        Parameters:
        - epsilon: Threshold for determining which rows to eliminate.
        - norm_abs: The norm type to use for calculation (default is 'euclidean').
        """
        # Normalize the matrix and RHS vector
        A_norm, b_norm, _ = normalize_features(self.A, self.b)
        num_rows, _ = A_norm.shape
        rows_to_delete = []

        # Identify rows to delete based on the norm criteria
        for i in range(num_rows):
            if self.opts.norm_abs == 'euclidean':
                row = A_norm.getrow(i)  # Extract the ith row as a sparse row
                row_norm = np.sqrt(np.sum(row.data ** 2))  # Compute the Euclidean norm of the row

                if row_norm / abs(b_norm[i]) < epsilon:
                    rows_to_delete.append(i)
            elif self.opts.norm_abs == 'abs':
                abs_coeff = np.abs(A_norm[i, :].toarray().flatten())
                if np.all(abs_coeff / abs(b_norm[i]) < epsilon):
                    rows_to_delete.append(i)
        # Modify the matrix and RHS vector by setting rows to zero
        A_modifiable = self.A.tolil()

        for i in rows_to_delete:
            A_modifiable[i, :] = 0
            self.b[i] = 0

        self.A = csr_matrix(A_modifiable)  # Convert back to CSR format
        self.change_dictionary["Eliminate Zero Rows"]["Rows"] = rows_to_delete

        # Count zero columns
        zero_columns = np.where(self.A.getnnz(axis=0) == 0)[0]
        print("Zero columns: ", len(zero_columns))


    def eliminate_zero_columns_operation(self, epsilon, norm_abs='euclidean'):
        """
        Eliminate columns in the matrix where the normalized column norm relative to the objective coefficients
        is below a given epsilon threshold.

        Parameters:
        - epsilon: Threshold for determining which columns to eliminate.
        - norm_abs: Norm type to use for calculation ('euclidean' or 'abs').
        """
        # Normalize features
        A_norm, _, _ = normalize_features(self.A, self.b)
        num_rows, num_cols = A_norm.shape
        columns_to_delete = []
        self.c = np.array(self.c)

        # Identify columns to delete based on the norm criteria
        for i in range(num_cols):
            if norm_abs == 'euclidean':
                column = self.A.toarray()[:, i]
                column_norm = np.linalg.norm(column)
            elif norm_abs == 'abs':
                column_norm = np.max(np.abs(self.A[:, i].toarray()))
            else:
                raise ValueError("Invalid norm_abs value. Use 'euclidean' or 'abs'.")

            # Avoid division by zero and handle small c[i]
            if abs(self.c[i]) > epsilon:
                criterion = column_norm / abs(self.c[i])
            else:
                criterion = column_norm

            # Decide whether to delete the column
            if criterion < epsilon:
                columns_to_delete.append(i)

        # Convert A to a modifiable format
        A_modifiable = self.A.tolil()

        # Adjust constraint senses based on the signs of A_ij in eliminated columns
        for row_index in range(A_modifiable.shape[0]):
            row = A_modifiable.getrow(row_index)
            coeffs = row[:, columns_to_delete].toarray().flatten()
            non_zero_coeffs = coeffs[coeffs != 0]

            if len(non_zero_coeffs) == 0:
                continue

            if np.all(non_zero_coeffs > 0):
                self.cons_senses[row_index] = '<='
            elif np.all(non_zero_coeffs < 0):
                self.cons_senses[row_index] = '>='

        # Zero out the columns
        for j in columns_to_delete:
            A_modifiable[:, j] = 0

        # Update the matrix A
        self.A = A_modifiable.tocsr()

        # Update objective coefficients for eliminated columns
        self.c[columns_to_delete] = 0

        # Print information about zero columns
        zero_columns_after = np.where(self.A.getnnz(axis=0) == 0)[0]
        print("Zero columns after:", len(zero_columns_after))

        # Record the columns that were zeroed out
        self.change_dictionary["Eliminate Zero Columns"] = {"Columns": columns_to_delete}

    def linear_dependency_rows(self, epsilon):
        basis_rows = []
        index_row = []
        # the new A is A with the colum of b
        self.b = np.array(self.b)
        self.A = self.A.toarray()
        new_A = np.vstack((self.A.T, self.b.reshape(1,-1)))
        new_A = new_A.T
        for i, fila in enumerate(new_A):
            if len(basis_rows) == 0:
                basis_rows.append(fila)
                index_row.append(i)
            else:
                # Crear una matriz con las filas base actuales como columnas
                base_matrix = np.vstack(basis_rows).T  # Cada columna es una fila base
                try:
                    # Resolver el sistema base_matrix * coef = fila
                    coeficientes, residuals, rango, _ = np.linalg.lstsq(base_matrix, fila, rcond=None)
                    if residuals.size > 0:
                        residual_norm = np.sqrt(residuals[0])
                    else:
                        # Calcular la norma del residual manualmente si residuals está vacío
                        residual = fila - base_matrix @ coeficientes
                        residual_norm = np.linalg.norm(residual)
                    # Si la norma del residual es mayor que la tolerancia, agregar la fila a la base
                    if residual_norm > epsilon:
                        basis_rows.append(fila)
                        index_row.append(i)
                except np.linalg.LinAlgError:
                    basis_rows.append(fila)
                    index_row.append(i)
        # The rows that are not in the basis put there coeeficients to zero
        print(len(index_row))
        for i in range(len(self.A)):
            if i not in index_row:
                self.A[i] = 0
                self.b[i] = 0
        self.A = csr_matrix(self.A)
        self.change_dictionary["Linear Dependency Rows"]["Rows"] = index_row

    def find_basis(self, epsilon=1e-2):

        self.b = np.array(self.b)
        self.A = self.A.toarray()
        new_A = np.vstack((self.A.T, self.b.reshape(1,-1)))
        new_A = new_A.T
        Q, R, P = qr(new_A.T, pivoting=True)  # shape of R: (n_rows, n_rows)

        # The diagonal entries of R correspond to the pivoted columns in Q.
        # In the row sense, we check how many of those are above epsilon.
        diag_R = np.abs(np.diag(R))
        rank = np.sum(diag_R > epsilon)

        # The first `rank` elements in P are the indices of the linearly independent rows.
        pivot_rows = P[:rank]

        # Zero out the rows that are not pivot rows
        # (This is analogous to what you do in your code.)
        # Make a copy or modify in place as needed
        A_out = self.A.copy()
        b_out = self.b.copy()
        for i in range(A_out.shape[0]):
            if i not in pivot_rows:
                A_out[i, :] = 0
                b_out[i] = 0

        # Convert back to sparse if needed
        A_out_csr = csr_matrix(A_out)

        self.A = A_out_csr
        self.b = b_out

    def find_basis_columns(self, epsilon):
        Q, R, P = qr(self.A, pivoting=True)  # shape of R: (n_rows, n_rows)

        # The diagonal entries of R correspond to the pivoted columns in Q.
        # In the row sense, we check how many of those are above epsilon.
        diag_R = np.abs(np.diag(R))
        rank = np.sum(diag_R > epsilon)

        # The first `rank` elements in P are the indices of the linearly independent rows.
        pivot_rows = P[:rank]

        A_out = self.A.copy()

        for i in range(A_out.shape[0]):
            if i not in pivot_rows:
                A_out[:,i] = 0

        # Convert back to sparse if needed
        A_out_csr = csr_matrix(A_out)

        self.A = A_out_csr


class PresolvepsilonOperationsSparse:
    def __init__(self, model = None,
                 opts = None):

        """
        Initialize the class with the model and the parameters for the presolve operations.
        """

        self.model = model
        self.change_elements = True
        # Initialize placeholders for matrices and model components
        self.A = None
        self.b = None
        self.c = None
        self.co = None
        self.lb = None
        self.ub = None
        self.of_sense = None
        self.cons_senses = None
        self.constraint_names = None
        self.variable_names = None
        self.original_row_index = None
        self.original_column_index = None

        self.opts = opts

        # Initialize dictionary to track changes made by presolve operations
        self.change_dictionary = defaultdict(nested_dict)

        # Initialize table to track the number of variables and constraints at each operation
        self.operation_table = []

    def load_model_matrices(self, model):
        """
        Extract and load matrices and other components from the optimization model.
        """

        if self.model is None:
            raise ValueError("Model is not provided.")

        if self.opts.bounds:
            print("Calculating bounds")
            lb, ub_new = calculate_bounds(model)
            path = os.path.join(os.getcwd(), 'model_matrices.json')
            model_bounds = build_model_from_json(path)
            model_bounds.setParam('OutputFlag', 0)
            model_bounds.optimize()
            print('Model optimized: ', model_bounds.objVal)
        else:
            print("Using the model matrices")
            model_bounds = model
        self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names, self.constraint_names = (
            get_model_matrices(model_bounds))
        ub = np.array(self.ub)
        print('Finite upper: ', np.sum(np.isfinite(ub)), 'Total upper: ', len(ub))

    def orchestrator_presolve_operations(self, model, epsilon = 1e-6):

        self.load_model_matrices(model)
        if self.opts.operate_epsilon_rows:
            print("Eliminate zero rows operation")
            self.eliminate_zero_rows_operation_sparse(epsilon = epsilon, norm_abs = self.opts.norm_abs)

        if self.opts.operate_epsilon_cols:
            print("Eliminate zero columns operation")
            self.eliminate_zero_columns_operation_sparse(epsilon = epsilon, norm_abs = self.opts.norm_abs)

        if self.opts.sparsification:
            print("Sparsification operation")
            #self.sparsification_operation_sparse(epsilon = epsilon)
            self.improve_sparsity_new(epsilon = epsilon)

        return (self.A, self.b, self.c, self.lb, self.ub, self.of_sense, self.cons_senses, self.co, self.variable_names, self.constraint_names,
                self.change_dictionary, self.operation_table)

    def sparsification_operation_sparse(self, epsilon=1e-6):
        """
        Perform a sparsification operation on the model's sparse matrix `self.A`.
        Elements in the matrix that are deemed insignificant (based on `epsilon`)
        will be set to zero, improving matrix sparsity.

        Parameters:
            epsilon (float): Threshold below which matrix elements are set to zero. Default is 1e-6.

        Returns:
            None: Updates `self.A` in-place as a sparse CSR matrix.
        """
        A_norm1, b_norm,  _ = normalize_features_sparse(self.A.copy(), self.b.copy())
        ub = np.array(self.ub)  # Upper bounds
        lb = np.array(self.lb)  # Lower bounds

        b_norm = np.array(b_norm)
        A_norm = csr_matrix(self.A).copy()  # Sparse matrix in CSR format
        #print("Non-zero elements in A_norm: ", A_norm.nonzero())

        max_sparsification = A_norm1.copy()
        data = max_sparsification.data
        A_indices = max_sparsification.indices
        A_indptr = max_sparsification.indptr

        # Handle upper and lower bounds on non-zero elements
        neg_mask = data < 0  # Mask for negative values
        pos_mask = data > 0  # Mask for positive values


        max_activity_data = np.zeros_like(data)
        max_activity_data[neg_mask] = data[neg_mask] * ub[A_indices[neg_mask]]
        max_activity_data[pos_mask] = data[pos_mask] * lb[A_indices[pos_mask]]

        # Compute row sums
        row_sums = np.add.reduceat(max_activity_data, A_indptr[:-1])

        row_indices = np.repeat(np.arange(len(A_indptr) - 1), np.diff(A_indptr))
        adjusted_data = row_sums[row_indices] - max_activity_data
        delayed_sparsification = csr_matrix((adjusted_data, A_indices, A_indptr), shape=max_sparsification.shape)
        upper_bound_matrix = A_norm1.copy()  # Copy the structure of A_norm
        upper_bound_matrix.data[neg_mask] *= lb[A_norm.indices[neg_mask]]  # Multiply negative entries by lb
        upper_bound_matrix.data[pos_mask] *= ub[A_norm.indices[pos_mask]] # Multiply positive entries by ub. This two lines define a variable to be in its maximum value
        #print(max_activity_variable.toarray())
        #upper_bound_matrix = A_norm1.multiply(ub)
        upper_bound_matrix = upper_bound_matrix.tocsr()  # Ensure CSR format for consistent indexing
        max_activity_variable = delayed_sparsification
        equal_sign = np.sign(max_activity_variable.data) == np.sign(upper_bound_matrix.data)

        # Compute the epsilon matrix
        epsilon_matrix = np.where(
            (np.isfinite(max_activity_variable.data)) & (max_activity_variable.data != 0),#& (upper_bound_matrix.data != 0) & (np.isfinite(upper_bound_matrix.data)),
            np.abs(upper_bound_matrix.data/ max_activity_variable.data),
            np.inf
            )
        condition = epsilon_matrix < epsilon
        A_norm.data[condition] = 0
        A_norm.eliminate_zeros()
        # By row # Calculate the number of elements in each row that are set to zero
        # Difference between the number of non-zero elements before and after sparsification by row
        # row_diff = np.diff(A_norm.indptr)
        # #print(row_diff)
        # # Get the rows where all the elements are zero
        # # Convert to array to avoid issues with sparse matrices
        # A_epsilon = csr_matrix((condition.astype(int), A_indices, A_indptr), shape=A_norm1.shape)
        # rows_epsilon, cols_epsilon = A_epsilon.nonzero()

        # print("Rows with epsilon condition:", len(rows_epsilon))    
        # print("Columns with importance condition:", len(cols_epsilon))
        # zero_rows_sparse = np.where(A_norm.getnnz(axis=1) == 0)[0]
        # print("Zero rows: ", len(zero_rows_sparse))
        self.A = A_norm
    
    def improve_sparsity_new(self, epsilon=1e-6, delta=1e-6):
        """
        Perform a sparsification operation on the model's sparse matrix self.A.
        Elements in the matrix that are deemed insignificant (based on epsilon)
        will be set to zero, improving matrix sparsity.

        Parameters:
            epsilon (float): Threshold below which matrix elements are set to zero. Default is 1e-6.

        Returns:
            None: Updates self.A in-place as a sparse CSR matrix.
        """
        A_norm1, b_norm,  _ = normalize_features_sparse(self.A.copy(), self.b.copy())
        num_rows, num_cols = A_norm1.shape
        ub = np.array(self.ub)  # Upper bounds
        lb = np.array(self.lb)  # Lower bounds
        b_norm = np.array(b_norm)
        A_norm = csr_matrix(self.A).copy()  # Sparse matrix in CSR format
        #print("Non-zero elements in A_norm: ", A_norm.nonzero())

        max_sparsification = A_norm1.copy()
        data = max_sparsification.data
        A_indices = max_sparsification.indices
        A_indptr = max_sparsification.indptr

        # Handle upper and lower bounds on non-zero elements
        neg_mask = data < 0  # Mask for negative values
        pos_mask = data > 0  # Mask for positive values

        max_activity_data = np.zeros_like(data)
        max_activity_data[neg_mask] = data[neg_mask] * ub[A_indices[neg_mask]]
        max_activity_data[pos_mask] = data[pos_mask] * lb[A_indices[pos_mask]]

        # Compute row sums
        row_sums = np.add.reduceat(max_activity_data, A_indptr[:-1])

        row_indices = np.repeat(np.arange(len(A_indptr) - 1), np.diff(A_indptr))
        adjusted_data = row_sums[row_indices] - max_activity_data
        delayed_sparsification = csr_matrix((adjusted_data, A_indices, A_indptr), shape=max_sparsification.shape)
        upper_bound_matrix = A_norm1.copy()  # Copy the structure of A_norm
        upper_bound_matrix.data[neg_mask] *= lb[A_norm.indices[neg_mask]]  # Multiply negative entries by lb
        upper_bound_matrix.data[pos_mask] *= ub[A_norm.indices[pos_mask]] # Multiply positive entries by ub. This two lines define a variable to be in its maximum value
        #print(max_activity_variable.toarray())
        #upper_bound_matrix = A_norm1.multiply(ub)
        upper_bound_matrix = upper_bound_matrix.tocsr()  # Ensure CSR format for consistent indexing
        max_activity_variable = delayed_sparsification

        # Compute the epsilon matrix
        epsilon_matrix = np.where(
            (np.isfinite(max_activity_variable.data)) & (max_activity_variable.data != 0), #& (upper_bound_matrix.data != 0) & (np.isfinite(upper_bound_matrix.data)),
            np.abs(upper_bound_matrix.data/ max_activity_variable.data),
            np.inf
            )
        
        condition_epsilon = epsilon_matrix < epsilon  # Shape: (nnz,)
  
        original_rows, original_cols = A_norm.nonzero()  # Indices of all nonzero values in A_norm1
        rows_epsilon = original_rows[condition_epsilon]  # Extract rows that meet the condition
        cols_epsilon = original_cols[condition_epsilon]

        vars_names_epsilon = [self.variable_names[col] for col in cols_epsilon]
        rows_names_epsilon = [self.constraint_names[row] for row in rows_epsilon]   
        unique_rows_epsilon = np.unique(rows_epsilon)
        unique_cols_epsilon = np.unique(cols_epsilon)
        print('Rows and cols with epsilon', len(unique_rows_epsilon), len(unique_cols_epsilon))
        # for every row get the vars_epsilon that are in the row
        vars_epsilon = defaultdict(list)
        A_norm.data[condition_epsilon] = 0
        self.A = A_norm  
        # self.A.eliminate_zeros() 
        # Efficiently populate vars_epsilon dictionary
        for row, col in zip(rows_epsilon, cols_epsilon):  
            vars_epsilon[row].append(col)  
        vars_epsilon = dict(vars_epsilon)
  
        new_model = build_model(self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names, self.constraint_names)
        model_flex = flexibility_constraints(new_model, self.constraint_names, vars_epsilon)
        self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names, self.constraint_names = get_model_matrices(model_flex)

    def improve_sparsity(self, epsilon=1e-6, delta=1e-6):
        """
        Perform a sparsification operation on the model's sparse matrix self.A.
        Elements in the matrix that are deemed insignificant (based on epsilon)
        will be set to zero, improving matrix sparsity.

        Parameters:
            epsilon (float): Threshold below which matrix elements are set to zero. Default is 1e-6.

        Returns:
            None: Updates self.A in-place as a sparse CSR matrix.
        """
        A_norm1, b_norm,  _ = normalize_features_sparse(self.A.copy(), self.b.copy())
        num_rows, num_cols = A_norm1.shape
        ub = np.array(self.ub)  # Upper bounds
        lb = np.array(self.lb)  # Lower bounds
        b_norm = np.array(b_norm)
        A_norm = csr_matrix(self.A).copy()  # Sparse matrix in CSR format
        #print("Non-zero elements in A_norm: ", A_norm.nonzero())

        max_sparsification = A_norm1.copy()
        data = max_sparsification.data
        A_indices = max_sparsification.indices
        A_indptr = max_sparsification.indptr

        # Handle upper and lower bounds on non-zero elements
        neg_mask = data < 0  # Mask for negative values
        pos_mask = data > 0  # Mask for positive values

        max_activity_data = np.zeros_like(data)
        max_activity_data[neg_mask] = data[neg_mask] * ub[A_indices[neg_mask]]
        max_activity_data[pos_mask] = data[pos_mask] * lb[A_indices[pos_mask]]

        # Compute row sums
        row_sums = np.add.reduceat(max_activity_data, A_indptr[:-1])

        row_indices = np.repeat(np.arange(len(A_indptr) - 1), np.diff(A_indptr))
        adjusted_data = row_sums[row_indices] - max_activity_data
        delayed_sparsification = csr_matrix((adjusted_data, A_indices, A_indptr), shape=max_sparsification.shape)
        # upper_bound_matrix = A_norm1.copy()  # Copy the structure of A_norm
        # upper_bound_matrix.data[neg_mask] *= lb[A_norm.indices[neg_mask]]  # Multiply negative entries by lb
        # upper_bound_matrix.data[pos_mask] *= ub[A_norm.indices[pos_mask]] # Multiply positive entries by ub. This two lines define a variable to be in its maximum value
        #print(max_activity_variable.toarray())
        upper_bound_matrix = A_norm1.multiply(ub)
        upper_bound_matrix = upper_bound_matrix.tocsr()  # Ensure CSR format for consistent indexing
        max_activity_variable = delayed_sparsification
        equal_sign = np.sign(max_activity_variable.data) == np.sign(upper_bound_matrix.data)

        # Compute the epsilon matrix
        epsilon_matrix = np.where(
            (np.isfinite(max_activity_variable.data)) & (max_activity_variable.data != 0), #& (upper_bound_matrix.data != 0) & (np.isfinite(upper_bound_matrix.data)),
            np.abs(upper_bound_matrix.data/ max_activity_variable.data),
            np.inf
            )

        min_sparsification = A_norm1.copy()
        min_activity_data = np.zeros_like(data)
        min_activity_data[neg_mask] = data[neg_mask] * lb[A_indices[neg_mask]]
        min_activity_data[pos_mask] = data[pos_mask] * ub[A_indices[pos_mask]]
        row_sums = np.add.reduceat(min_activity_data, A_indptr[:-1])
        adjusted_data_min = row_sums[row_indices] - min_activity_data
        lower_bound_matrix = A_norm1.multiply(lb)
        # lower_bound_matrix.data[neg_mask] *= ub[A_norm.indices[neg_mask]]  # Multiply negative entries by lb
        # lower_bound_matrix.data[pos_mask] *= lb[A_norm.indices[pos_mask]] # Multiply positive entries by ub. This two lines define a variable to be in its maximum value
        min_activity_variable = csr_matrix((adjusted_data_min, A_indices, A_indptr), shape=min_sparsification.shape)
        importance_matrix = np.where(min_activity_variable.data != 0 & np.isfinite(min_activity_data),
                                    np.abs(lower_bound_matrix.data / min_activity_variable.data),
                                    0)

        condition_epsilon = epsilon_matrix < epsilon  # Shape: (nnz,)
        original_rows, original_cols = A_norm1.nonzero()  # Indices of all nonzero values in A_norm1

        # Step 2: Apply epsilon condition to filter only valid indices
        rows_epsilon = original_rows[condition_epsilon]  # Extract rows that meet the condition
        cols_epsilon = original_cols[condition_epsilon]

        print(len(rows_epsilon))
        print(len(cols_epsilon))

        # print("Number of elements below epsilon: ", np.sum(condition_epsilon))
        # condition_importance = importance_matrix > delta  # Shape: (nnz,))
        A_norm.data[condition_epsilon] = 0
        A_epsilon = csr_matrix(
            (A_norm.data, A_indices, A_indptr), shape=A_norm1.shape
        )
        # A_importance = csr_matrix(
        #     (condition_importance.astype(int), A_indices, A_indptr), shape=A_norm1.shape
        # )
        # # # Retrieve row, col indices efficiently
        # rows_epsilon, cols_epsilon = A_epsilon.nonzero()
        # rows_importance, cols_importance = A_importance.nonzero()
        # print("Rows with epsilon condition:", len(rows_epsilon))    
        # print("Columns with importance condition:", len(cols_epsilon))
        
        # #self.A = A_epsilon
        # #Use defaultdict(set) for efficient dictionary handling
        # epsilon_dict, importance_dict = defaultdict(set), defaultdict(set)

        # for row, col in zip(rows_epsilon, cols_epsilon):
        #     epsilon_dict[col].add(row)

        # for row, col in zip(rows_importance, cols_importance):
        #     importance_dict[col].add(row)

        # # Find columns where both conditions exist
        # columns_with_both_conditions = set(epsilon_dict.keys()) & set(importance_dict.keys())

        # # Classify per variable the rows where the variable appears
        # total_rows, total_cols = A_norm1.nonzero()
        variables_rows = defaultdict(set)
        # columns_only_epsilon = set(epsilon_dict.keys()) - columns_with_both_conditions
        rows_variables = defaultdict(set)

        # columns_only_epsilon = set(epsilon_dict.keys()) - columns_with_both_conditions
        # print("Columns only with epsilon condition:", len(columns_only_epsilon))
        valid_cols = set(cols_epsilon)  # Convert to a set for O(1) lookups

        for row, col in zip(*A_norm1.nonzero()):  # Avoid unpacking twice
            if col in valid_cols:  # Fast membership check
                variables_rows[col].add((row, self.constraint_names[row]))
                rows_variables[row].add((col, self.variable_names[col]))

        # print("Both conditions hold for columns:", len(rows_variables))

        # rows_to_delete = set()
        # for col in columns_with_both_conditions:
        #     if importance_dict[col]:  # Ensure the set is not empty
        #         #print("Importance rows in col", col, ":", importance_dict[col])
        #         rows_to_delete.add(importance_dict[col].pop())

        print(variables_rows)
        # print("Rows marked for deletion:", len(rows_to_delete)) 
        self.cons_senses, new_vars = flexibility_constraints(self.A, self.b, self.cons_senses, self.constraint_names)
        # # # In A only delete the A epsilon corresponding to the self cons change
        # self.A = A_epsilon
        #self.A.eliminate_zeros()
        # rows_to_delete = list(map(int, rows_to_delete))
        # mask_rows_to_delete = np.ones(num_rows, dtype=bool)
        # mask_rows_to_delete[rows_to_delete] = False
        # self.b = np.array(self.b)
        # if len(rows_to_delete) > 0:
        #     mask = mask_rows_to_delete
        #     # Number of true values in the mask
        #     print("Number of true values in the mask: ", np.sum(mask))
        #     self.A = csr_matrix(spdiags(mask.astype(int), 0, num_rows, num_rows).dot(self.A))
        #     self.b[rows_to_delete] = 0

        # columns_to_delete = list(map(int, new_vars))
        # mask_cols = np.ones(num_cols, dtype=bool)
        # mask_cols[columns_to_delete] = False

        # # Create a sparse diagonal mask for columns
        # mask_diag_cols = spdiags(mask_cols.astype(int), 0, num_cols, num_cols)

        # # Multiply from the right to zero out selected columns
        # self.A = self.A.dot(mask_diag_cols)

    def eliminate_zero_rows_operation_sparse(self, epsilon, norm_abs='euclidean'):
        """
        Eliminate rows in the matrix where the normalized row norm relative to the RHS vector
        is below a given epsilon threshold.

        Parameters:
        - epsilon: Threshold for determining which rows to eliminate.
        - norm_abs: Norm type to use for calculation ('euclidean' or 'abs').
        """
        # Normalize the matrix and RHS vector
        A_norm, b_norm, _ = normalize_features_sparse(self.A.copy(), self.b.copy())
        num_rows, _ = A_norm.shape
        self.b = np.array(self.b)

        # Compute norms based on the specified method
        if norm_abs == 'euclidean':
            row_sums_of_squares = A_norm.power(2).sum(axis=1)
            row_norms = np.sqrt(np.asarray(row_sums_of_squares).ravel())
        elif norm_abs == 'abs':
            row_norms = np.asarray(A_norm.max(axis=1).toarray()).flatten()
        else:
            raise ValueError("Invalid norm_abs value. Use 'euclidean' or 'abs'.")

        # Compute ratio of row norm to the absolute value of b_norm
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(b_norm != 0, row_norms / np.abs(b_norm), np.inf)

        # Identify rows to delete
        rows_to_delete = np.where(ratio < epsilon)[0].tolist()

        # Debugging information
        print("Rows to delete: ", len(rows_to_delete))

        # Modify the matrix and RHS vector by zeroing out rows to delete
        if rows_to_delete:
            mask = np.ones(num_rows, dtype=bool)
            mask[rows_to_delete] = False
            self.A = csr_matrix(spdiags(mask.astype(int), 0, num_rows, num_rows).dot(self.A_norm))
            self.b[rows_to_delete] = 0

        # Store changes
        self.change_dictionary["Eliminate Zero Rows"]["Rows"] = rows_to_delete

        # Count zero columns
        zero_columns = np.where(self.A.getnnz(axis=0) == 0)[0]
        print("Zero columns: ", len(zero_columns))


    def eliminate_zero_columns_operation_sparse(self, epsilon, norm_abs='euclidean'):
        """
        Eliminate columns in the matrix where the normalized column norm relative to the objective coefficients
        is below a given epsilon threshold (sparse implementation).

        Parameters:
        - epsilon: Threshold for determining which columns to eliminate.
        - norm_abs: Norm type to use for calculation ('euclidean' or 'abs').
        """
        # Normalize features (A_norm is sparse)
        #A_norm, c_norm, _ = normalize_features_sparse(self.A.T.copy(), self.c.copy())
        A_norm = self.A.copy()
        c_norm = self.c.copy()
        self.c = np.array(c_norm)
        num_rows, num_cols = self.A.shape
        print("Number of columns: ", num_cols)
        # Compute norms based on the specified method
        if norm_abs == 'euclidean':
            col_sums_of_squares = A_norm.power(2).sum(axis=0)
            column_norms = np.sqrt(np.asarray(col_sums_of_squares).ravel())
            print(len(np.where(column_norms < 1)[0]))
        elif norm_abs == 'abs':
            column_norms = abs(A_norm).max(axis=0).toarray().flatten()
            print(len(np.where(column_norms < 1)[0]))
        else:
            raise ValueError("Invalid norm_abs value. Use 'euclidean' or 'abs'.")

        # Compute ratio of column norm to the absolute value of c_norm
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_columns = np.where(np.abs(c_norm) != 0, column_norms /np.abs(c_norm), column_norms)

        print("Ratio columns: ", len(np.where(ratio_columns < epsilon)[0]))
        # Identify columns to delete
        columns_to_delete = np.where(ratio_columns < epsilon)[0]
        # Adjust constraint senses based on eliminated columns
        for row_index in range(num_rows):
            # Extract this row as a sparse vector
            row = self.A.getrow(row_index)
            sub_indices = np.in1d(row.indices, columns_to_delete)  # indices within the row that correspond to columns_to_delete
            sub_data = row.data[sub_indices]

            if len(sub_data) == 0:
                continue
            if np.all(sub_data > 0):
                self.cons_senses[row_index] = '<='
            elif np.all(sub_data < 0):
                self.cons_senses[row_index] = '>='
            else:
                pass

        mask_cols = np.ones(num_cols, dtype=bool)
        mask_cols[columns_to_delete] = False

        # Create a sparse diagonal mask for columns
        mask_diag_cols = spdiags(mask_cols.astype(int), 0, num_cols, num_cols)

        # Multiply from the right to zero out selected columns
        self.A = self.A.dot(mask_diag_cols)

        # Update self.c if needed (e.g., zero out the objective coefficients for eliminated columns)
        self.c[columns_to_delete] = 0

        # Print information about zero columns
        zero_columns_after = np.where(self.A.getnnz(axis=0) == 0)[0]
        print("Zero columns after:", len(zero_columns_after))

        # Record the columns that were zeroed out
        self.change_dictionary["Eliminate Zero Columns"] = {"Columns": columns_to_delete.tolist()}