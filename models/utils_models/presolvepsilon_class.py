import numpy as np
from scipy.sparse import csr_matrix, spdiags
from utils_models.utils_functions import get_model_matrices, \
    find_corresponding_negative_rows_with_indices, nested_dict, normalize_features_sparse, normalize_features, \
    matrix_sparsification, calculate_bounds, build_dual_model_from_json, build_model_from_json, save_json, build_model
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime
import logging
import os 
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

# logging.basicConfig()
# log = logging.getLogger(__name__)
# log.setLevel('INFO')


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
        self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names = (
            get_model_matrices(model_bounds))
        ub = np.array(self.ub)
        print('Finite upper: ', np.sum(np.isfinite(ub)), 'Total upper: ', len(ub))

    def orchestrator_presolve_operations(self, model, epsilon = 1e-6):

        self.load_model_matrices(model)
        if self.opts.operate_epsilon_rows:
            print("Eliminate zero rows operation")
            self.eliminate_zero_rows_operation_sparse(epsilon = epsilon)

        if self.opts.operate_epsilon_cols:
            print("Eliminate zero columns operation")
            self.eliminate_zero_columns_operation_sparse(epsilon = epsilon)
        
        if self.opts.sparsification:
            print("Sparsification operation")
            self.sparsification_operation(epsilon = epsilon)
        
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
        # Convert
        A_modifiable = A_dense.copy()
        max_sparsification = np.where(A_norm1 < 0, A_norm1* self.ub, np.where(A_norm1 > 0, A_norm1 * self.lb, 0))
        ones = np.ones((A_dense.shape[1], A_dense.shape[1]))
        np.fill_diagonal(ones, 0)
        max_activity_variable = max_sparsification @ ones
        upper_bound_matrix = A_norm1 * self.ub
        # print(upper_bound_matrix[2,0])
        # print(max_activity_variable[2,0])
        # only divide upper_bound / max_activity if finite values are present (!= 0)
        epsilon_matrix = np.where(
                        (np.isfinite(max_activity_variable)) & (upper_bound_matrix != 0) & (np.isfinite(upper_bound_matrix)),
                        #np.abs(upper_bound_matrix) / np.abs(max_activity_variable),
                        np.abs(upper_bound_matrix / max_activity_variable),
                        np.inf
                    )

        # Count the number of elements that are less than epsilon
        indices = np.where(epsilon_matrix < epsilon)
        # print the values of the indices
        # View if the indices are not zero
        # # set the elements to zero in modifiable matrix
        A_modifiable[indices] = 0
        # How many elements are 0 
        self.A = csr_matrix(A_modifiable)   
        # Build the model
        # model = build_model(self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names)
        # model.setParam('OutputFlag', 0)
        # model.optimize()
        # print('Model optimized: ', model.objVal)

    # def eliminate_zero_rows_operation1(self, epsilon, norm_abs = 'euclidean'):
    #     A_norm, b_norm, _ = normalize_features(self.A, self.b)
    #     num_rows, num_cols = A_norm.shape  
    #     rows_to_delete = []
        
    #     # Identify rows to be marked based on the norm criteria
    #     for i in range(num_rows):
    #     # Extract the ith row as a sparse row
    #         row = A_norm.getrow(i)
            
    #         # Compute the norm of the row directly from its nonzero values
    #         # row.data returns a 1D array of nonzero values in this row.
    #         row_norm = np.sqrt(np.sum(row.data ** 2))
            
    #         # Check if the normalized row norm compared to b_norm[i] is below epsilon
    #         if row_norm / abs(b_norm[i]) < epsilon:
    #             rows_to_delete.append(i)

    #         if np.all(abs_coeff / abs(b_norm[i]) < epsilon):
    #         #     rows_to_delete.append(i)

    #     #print("Rows to delete: ", len(rows_to_delete))
    #     #print(rows_to_delete)
    #     # rows_to_keep = []
    #     # for i in rows_to_delete:
    #     #     if self.b[i] <= 0:
    #     #         delete = True
    #     #     else:
    #     #         rows_to_keep.append(i)
    #     #         # warnings.warn(f"Model is infeasible due to a non-redundant zero row at index {i}.")

    #     # # Update rows_to_delete based on rows_to_keep
    #     # rows_to_delete1 = [i for i in rows_to_delete if i not in rows_to_keep]

    #     #print(rows_to_delete)
    #     # Convert the matrix to a format that supports assignment, like COO or LIL
    #     A_modifiable = self.A.tolil()
    #     for i in rows_to_delete:
    #         A_modifiable[i, :] = 0  # Set the entire row to zero
    #         self.b[i] = 0
    #     # Convert back to CSR format

    #     self.A = csr_matrix(A_modifiable)
    #     self.change_dictionary["Eliminate Zero Rows"]["Rows"] = rows_to_delete
    #     # Solve the problem again, construct the model and optimize it
    #     # Count also the numbers of zero columns and rows
    #     # Count the number of zero columns
        
    #     zero_columns = np.where(self.A.getnnz(axis=0) == 0)[0]
    #     #zero_rows = np.where(~A_norm.any(axis=1))[0]
    #     print("Zero columns: ", len(zero_columns))


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
            self.A = csr_matrix(spdiags(mask.astype(int), 0, num_rows, num_rows).dot(self.A))
            self.b[rows_to_delete] = 0

        # Store changes
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

    def eliminate_zero_columns_operation_sparse(self, epsilon, norm_abs='euclidean'):
        """
        Eliminate columns in the matrix where the normalized column norm relative to the objective coefficients
        is below a given epsilon threshold (sparse implementation).

        Parameters:
        - epsilon: Threshold for determining which columns to eliminate.
        - norm_abs: Norm type to use for calculation ('euclidean' or 'abs').
        """
        # Normalize features (A_norm is sparse)
        A_norm, c_norm, _ = normalize_features_sparse(self.A.T.copy(), self.c.copy())
        A_norm = self.A.copy()
        c_norm = self.c.copy()
        self.c = np.array(self.c)
        num_rows, num_cols = A_norm.shape

        # Compute norms based on the specified method
        if norm_abs == 'euclidean':
            col_sums_of_squares = A_norm.power(2).sum(axis=0)
            column_norms = np.sqrt(np.asarray(col_sums_of_squares).ravel())
        elif norm_abs == 'abs':
            column_norms = np.asarray(A_norm.max(axis=0).toarray()).flatten()
        else:
            raise ValueError("Invalid norm_abs value. Use 'euclidean' or 'abs'.")

        # Compute ratio of column norm to the absolute value of c_norm
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_columns = np.where(np.abs(c_norm) != 0, column_norms / np.abs(c_norm), column_norms)

        # Identify columns to delete
        columns_to_delete = np.where(ratio_columns < epsilon)[0]

        # Debugging information
        print("Columns to delete: ", len(columns_to_delete))

        # Adjust constraint senses based on eliminated columns
        for row_index in range(num_rows):
            row = self.A.getrow(row_index)
            sub_indices = np.in1d(row.indices, columns_to_delete)
            sub_data = row.data[sub_indices]

            if len(sub_data) == 0:
                continue

            if np.all(sub_data > 0):
                self.cons_senses[row_index] = '<='
            elif np.all(sub_data < 0):
                self.cons_senses[row_index] = '>='

        # Zero out the selected columns without removing them
        mask_cols = np.ones(num_cols, dtype=bool)
        mask_cols[columns_to_delete] = False

        mask_diag_cols = spdiags(mask_cols.astype(int), 0, num_cols, num_cols)
        self.A = self.A.dot(mask_diag_cols)

        # Update objective coefficients for eliminated columns
        self.c[columns_to_delete] = 0

        # Print information about zero columns
        zero_columns_after = np.where(self.A.getnnz(axis=0) == 0)[0]
        print("Zero columns after:", len(zero_columns_after))

        # Record the columns that were zeroed out
        self.change_dictionary["Eliminate Zero Columns"] = {"Columns": columns_to_delete.tolist()}