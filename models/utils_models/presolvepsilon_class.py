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
                 eliminate_zero_rows_epsilon = None,
                 sparsification_bool = None,
                 opts = None):

        """
        Initialize the class with the model and the parameters for the presolve operations.
        """

        self.model = model
                # input data for some operations
        self.eliminate_zero_rows_epsilon = eliminate_zero_rows_epsilon
        self.sparsification = sparsification_bool
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
        # Generate original row indices from 0 to the number of constraints
        #self.original_row_index = list(range(self.A.A.shape[0]))

        # Generate original columns indices from 0 to the number of variables
        #self.original_column_index = list(range(self.A.A.shape[1]))

        # Add initial counts to the operation table
        #self.operation_table.append(("Initial", len(self.cons_senses), len(self.variable_names), self.A.nnz))

    def orchestrator_presolve_operations(self, model, epsilon = 1e-6):

        self.load_model_matrices(model)
        if self.opts.operate_epsilon_rows:
            print("Eliminate zero rows operation")
            self.eliminate_zero_rows_operation_sparse(epsilon = epsilon)
        
        if self.opts.sparsification:
            print("Sparsification operation")
            self.sparsification_operation(epsilon = epsilon)
            

        if self.opts.operate_epsilon_cols:
            print("Eliminate zero columns operation")
            self.eliminate_zero_columns_operation_sparse(epsilon = epsilon)
        
        if self.opts.change_elements:
            print("Change elements operation")
            self.change_elements_operation()

        return (self.A, self.b, self.c, self.lb, self.ub, self.of_sense, self.cons_senses, self.co, self.variable_names,
                self.change_dictionary, self.operation_table)  
    
    def change_elements_operation(self):
        """
        Change elements in the model matrices. Sparse matrix
        """
        # Initialize the dictionary to track changes made by the operation
        self.change_dictionary["Change Elements"] = defaultdict(list)

        if not isinstance(self.A, lil_matrix):
            self.A = self.A.tolil()

        # Modify the element (0, 0) of the sparse matrix
        self.A[0, 0] = 0

        # Record the changes in the dictionary
        self.change_dictionary["Change Elements"]["Rows"].append(0)
        self.change_dictionary["Change Elements"]["Columns"].append(0)
        self.change_dictionary["Change Elements"]["Values"].append(0)

        # Optionally convert back to CSR format for efficient computation
        self.A = self.A.tocsr()


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

    def eliminate_zero_rows_operation(self, epsilon):
        A_norm, b_norm, _ = normalize_features(self.A, self.b)
        num_rows, num_cols = A_norm.shape  
        rows_to_delete = []
        
        # Identify rows to be marked based on the norm criteria
        for i in range(num_rows):
        # Extract the ith row as a sparse row
            row = A_norm.getrow(i)
            
            # Compute the norm of the row directly from its nonzero values
            # row.data returns a 1D array of nonzero values in this row.
            row_norm = np.sqrt(np.sum(row.data ** 2))
            
            # Check if the normalized row norm compared to b_norm[i] is below epsilon
            if row_norm / abs(b_norm[i]) < epsilon:
                rows_to_delete.append(i)

            # if np.all(abs_coeff / abs(b_norm[i]) < epsilon):
            #     rows_to_delete.append(i)

        #print("Rows to delete: ", len(rows_to_delete))
        #print(rows_to_delete)
        # rows_to_keep = []
        # for i in rows_to_delete:
        #     if self.b[i] <= 0:
        #         delete = True
        #     else:
        #         rows_to_keep.append(i)
        #         # warnings.warn(f"Model is infeasible due to a non-redundant zero row at index {i}.")

        # # Update rows_to_delete based on rows_to_keep
        # rows_to_delete1 = [i for i in rows_to_delete if i not in rows_to_keep]

        #print(rows_to_delete)
        # Convert the matrix to a format that supports assignment, like COO or LIL
        A_modifiable = self.A.tolil()
        for i in rows_to_delete:
            A_modifiable[i, :] = 0  # Set the entire row to zero
            self.b[i] = 0
        # Convert back to CSR format

        self.A = csr_matrix(A_modifiable)
        self.change_dictionary["Eliminate Zero Rows"]["Rows"] = rows_to_delete
        # Solve the problem again, construct the model and optimize it
        # Count also the numbers of zero columns and rows
        # Count the number of zero columns
        
        zero_columns = np.where(self.A.getnnz(axis=0) == 0)[0]
        #zero_rows = np.where(~A_norm.any(axis=1))[0]
        print("Zero columns: ", len(zero_columns))

    def eliminate_zero_rows_operation_sparse(self, epsilon):
        """
        Eliminates (sets to zero) rows in a sparse matrix A and vector b 
        where the norm of the row is less than epsilon relative to the corresponding 
        value in b.

        Parameters:
        epsilon (float): Threshold value to identify rows to zero out.

        Returns:
        None: The operation modifies self.A and self.b in place.
        """
        # Normalize features to obtain normalized A and b
        A_norm, b_norm, _ = normalize_features_sparse(self.A.copy(), self.b.copy())
        
        # Ensure b is a 1D NumPy array
        self.b = np.array(self.b)
        
        # Compute the sum of squares of each row (norm squared)
        row_sums_of_squares = A_norm.power(2).sum(axis=1)

        # Convert to 1D NumPy array
        row_sums_of_squares = np.asarray(row_sums_of_squares).ravel()

        # Compute the Euclidean norm for each row
        row_norms = np.sqrt(row_sums_of_squares)

        # Compute ratio of row_norm to the absolute value of b_norm
        # Handle potential divide-by-zero issues
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(b_norm != 0, row_norms / np.abs(b_norm), np.inf)

        # Identify rows where ratio is less than epsilon
        rows_to_delete = np.where(ratio < epsilon)[0].tolist()
        
        # Debugging information about rows being eliminated
        print("Rows to delete: ", len(rows_to_delete))
        if len(rows_to_delete) > 0:
            # Create a mask to keep only the rows not in rows_to_delete
            mask = np.ones(self.A.shape[0], dtype=bool)
            mask[rows_to_delete] = False

            # Create a sparse diagonal matrix to apply the mask
            mask_diag = spdiags(mask.astype(int), 0, mask.size, mask.size)

            # Apply the mask to zero out the rows in A
            self.A = mask_diag.dot(self.A)

            # Set the corresponding entries in b to zero
            self.b[rows_to_delete] = 0
    
    def eliminate_zero_columns_operation(self, epsilon):
        # Copy the original matrix for reference
        
        # Normalize features if necessary (assuming normalize_features doesn't modify self.A in-place)
        A_norm, _, _ = normalize_features(self.A, self.b)
        num_rows, num_cols = A_norm.shape
        columns_to_delete = []

        # Identify columns to delete based on the norm criteria
        for i in range(num_cols):
            column = self.A.toarray()[:, i]
            column_norm = np.linalg.norm(column)
            
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
            # Get coefficients in the columns to delete
            coeffs = row[:, columns_to_delete].toarray().flatten()
            # Filter out zero coefficients
            non_zero_coeffs = coeffs[coeffs != 0]
            
            if len(non_zero_coeffs) == 0:
                # No non-zero coefficients, no change in sense
                continue
            
            # Check if all coefficients have the same sign
            if np.all(non_zero_coeffs > 0):
                # All positive coefficients, set sense to '<='
                self.cons_senses[row_index] = '<='
            elif np.all(non_zero_coeffs < 0):
                # All negative coefficients, set sense to '>='
                self.cons_senses[row_index] = '>='
            else:
                # Mixed signs, cannot determine a single sense
                # Decide based on the problem context or keep the original sense
                # Delete the column from columns_to_delete
                pass
                #columns_to_delete = [col for col in columns_to_delete if col not in columns_to_delete]
                #   # Alternatively, you can raise a warning or handle this case as needed
                
        # Zero out the columns
        for j in columns_to_delete:
            A_modifiable[:, j] = 0
        
        # Update the matrix A
        self.A = A_modifiable.tocsr()
        
        # Optionally, update variable-related data structures
        # if hasattr(self, 'variable_names'):
        #     self.variable_names = [name for idx, name in enumerate(self.variable_names) if idx not in columns_to_delete]
        # if hasattr(self, 'variable_bounds'):
        #     self.variable_bounds = [bound for idx, bound in enumerate(self.variable_bounds) if idx not in columns_to_delete]


    def eliminate_zero_columns_operation_sparse(self, epsilon):
        # Normalize features (A_norm is sparse)
        A_norm, c_norm, _ = normalize_features_sparse(self.A.T.copy(), self.c.copy())
        A_norm = self.A.copy()
        num_rows, num_cols = A_norm.shape

        # Compute the sum of squares for each column
        # A_norm.power(2).sum(axis=0) efficiently computes column-wise sum of squares
        col_sums_of_squares = A_norm.power(2).sum(axis=0)
        col_sums_of_squares = np.asarray(col_sums_of_squares).ravel()
        
        # Compute the Euclidean norm of each column
        column_norms = np.sqrt(col_sums_of_squares)
        self.c = np.array(self.c)
        # Compute ratio similarly as done for rows, but now for columns.
        # Assuming `self.c` is a vector of objective coefficients for each column.
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_columns = np.where(np.abs(c_norm) != 0, column_norms / np.abs(c_norm), column_norms)

        # Identify columns to zero out based on epsilon
        columns_to_delete = np.where(ratio_columns < epsilon)[0]

        # If you need to adjust constraint senses based on eliminated columns, do so before zeroing them out.
        # For each row, check the sign of coefficients in the columns_to_delete.
        # This still requires a loop over rows, but we can do it in a sparse-friendly way:
        for row_index in range(num_rows):
            # Extract this row as a sparse vector
            row = self.A.getrow(row_index)
            
            # Extract coefficients in columns_to_delete
            # To do this efficiently, select the required columns from the row
            # E.g., sub_row = row[:, columns_to_delete]
            # For large sets of columns_to_delete, a direct indexing might be costly, consider a more efficient approach if needed.
            
            # Here, let's simply get the nonzero pattern of the row and filter:
            sub_indices = np.in1d(row.indices, columns_to_delete)  # indices within the row that correspond to columns_to_delete
            sub_data = row.data[sub_indices]

            if len(sub_data) == 0:
                # No nonzero coefficients in these columns, no change in sense
                continue

            # Check sign pattern
            if np.all(sub_data > 0):
                self.cons_senses[row_index] = '<='
            elif np.all(sub_data < 0):
                self.cons_senses[row_index] = '>='
            else:
                # Mixed signs: Handle according to your logic. 
                # Possibly no sense change or special handling here.
                pass

        # Now zero out the selected columns without removing them.
        # Create a mask for columns
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

        # Optionally, record the columns that were zeroed out
        self.change_dictionary["Eliminate Zero Columns"] = {"Columns": columns_to_delete.tolist()}
