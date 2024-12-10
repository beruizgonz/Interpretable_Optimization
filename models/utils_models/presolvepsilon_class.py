import numpy as np
from scipy.sparse import csr_matrix
from utils_models.utils_functions import get_model_matrices, \
    find_corresponding_negative_rows_with_indices, nested_dict, linear_dependency, normalize_features, \
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
            self.eliminate_zero_rows_operation(epsilon = epsilon)
        
        if self.opts.sparsification:
            print("Sparsification operation")
            self.sparsification_operation(epsilon = epsilon)
            

        if self.opts.operate_epsilon_cols:
            print("Eliminate zero columns operation")
            self.eliminate_zero_columns_operation(epsilon = epsilon)
        
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
        A_norm1, _, _ = normalize_features(A_norm, self.b)
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
        original_A = self.A.A.copy()
        A_norm, b_norm, _ = normalize_features(self.A, self.b)
        num_rows, num_cols = A_norm.shape  
        rows_to_delete = []
        
        # Identify rows to be marked based on the norm criteria
        for i in range(num_rows):
            row = A_norm.getrow(i)
            row_norm = np.linalg.norm(row.toarray())
            abs_coeff = np.abs(row.data)
            
            if row_norm / abs(self.b[i]) < epsilon:
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
    
    def eliminate_zero_columns_operation(self, epsilon):
        # Copy the original matrix for reference
        original_A = self.A.copy()
        
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

        