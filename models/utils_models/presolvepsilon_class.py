import numpy as np
from scipy.sparse import csr_matrix
from utils_models.utils_functions import get_model_matrices, \
    find_corresponding_negative_rows_with_indices, nested_dict, linear_dependency, normalize_features, \
    matrix_sparsification, calculate_bounds2, build_dual_model_from_json, build_model_from_json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime
import logging
import os 
from scipy.sparse import coo_matrix, csr_matrix

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
        
        lb, new = calculate_bounds2(model)
        path = os.path.join(os.getcwd(), 'model_matrices.json')
        model_bounds = build_model_from_json(path)
          
        self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names = (
            get_model_matrices(model_bounds))

        # Generate original row indices from 0 to the number of constraints
        self.original_row_index = list(range(self.A.A.shape[0]))

        # Generate original columns indices from 0 to the number of variables
        self.original_column_index = list(range(self.A.A.shape[1]))

        # Add initial counts to the operation table
        self.operation_table.append(("Initial", len(self.cons_senses), len(self.variable_names), self.A.nnz))

    def orchestrator_presolve_operations(self, model, epsilon = 1e-6):

        self.load_model_matrices(model)
        if self.opts.operate_epsilon_rows:
            self.eliminate_zero_rows_operation(epsilon = epsilon)
        
        if self.opts.sparsification:
            self.sparsification_operation(epsilon = epsilon)

        return (self.A, self.b, self.c, self.lb, self.ub, self.of_sense, self.cons_senses, self.co, self.variable_names,
                self.change_dictionary, self.operation_table)    


    def sparsification_operation(self,epsilon = 1e-6):
        """
        Perform sparsification operation on the model matrices.
        """
        A_norm, _, _ = normalize_features(self.A, self.b)
        self.ub = np.array(self.ub)

        # Covert the matrix to a CSR format
        A_norm = csr_matrix(A_norm)
        # Convert A_norm to an array
        A_dense = self.A.toarray()
        A_norm = A_norm.toarray()
        # Convert
        A_modifiable = self.A.tolil()
        max_sparsification = np.where(A_norm < 0, +A_norm* self.ub, np.where(A_norm > 0, A_norm * self.lb, 0))
        ones = np.ones((A_norm.shape[1], A_norm.shape[1]))
        np.fill_diagonal(ones, 0)
        max_activity_variable = max_sparsification @ ones
        upper_bound_matrix = A_norm * self.ub
        # print(upper_bound_matrix[2,0])
        # print(max_activity_variable[2,0])
        # only divide upper_bound / max_activity if finite values are present (!= 0)
        epsilon_matrix = np.where(
                        (max_activity_variable != 0) & (max_activity_variable != np.inf) & (max_activity_variable != -np.inf),
                        upper_bound_matrix / max_activity_variable,
                        np.inf
                    )
        #print(epsilon_matrix)
        # look for the indices of the elements that are less than epsilon
        indices = np.where(np.abs(epsilon_matrix) < epsilon)
        # # set the elements to zero in modifiable matrix
        A_modifiable[indices] = 0
        self.A = A_modifiable.tocsr()

        #for i in range(A_norm.shape[0]):
    
            # row = A_norm.getrow(i)  # Get the i-th row as a sparse matrix
            # row_data = row.data     # Non-zero values in the row
            # row_indices = row.indices  # Column indices corresponding to the non-zero values

            # # Create boolean masks for positive and negative coefficients
            # positive_mask = row_data > 0
            # negative_mask = row_data < 0

            # # Get the column indices of positive and negative coefficients
            # positive_coeff_indices = row_indices[positive_mask]
            # negative_coeff_indices = row_indices[negative_mask]

            # # If you need the actual coefficients:
            # positive_coeff_values = row_data[positive_mask]
            # negative_coeff_values = row_data[negative_mask]
            # #print(positive_coeff_indices)
            # lb = self.lb[positive_coeff_indices]
            # ub = self.ub[positive_coeff_indices]
            # print(lb)
            # print(lb.shape)
            # print(ub.shape)
            # print(positive_coeff_values.shape)
            # print(positive_coeff_indices.shape)
            # # Now you can proceed with your logic using these indices and values
        #     for idx, j in enumerate(positive_coeff_indices):
        #         # Access the positive coefficient value
        #         coeff_value = positive_coeff_values[idx]
        #         #Proceed with your logic for positive coefficients
        #         # Example:
        #         positive_bound = self.ub[j]
        #         cond = coeff_value / positive_bound < epsilon
        #         # Handle 'cond' accordingly
        #         if cond: 
        #             A_modifiable[i, j] = 0
        #             # self.change_dictionary["Sparsification"]["Rows"].append(i)
        #             # self.change_dictionary["Sparsification"]["Columns"].append(j)
        #             # self.change_dictionary["Sparsification"]["Values"].append(coeff_value)
        #             # self.change_dictionary["Sparsification"]["Bounds"].append(positive_bound)

        #     for idx, j in enumerate(negative_coeff_indices):
        #         # Access the negative coefficient value
        #         coeff_value = negative_coeff_values[idx]
        #         # Proceed with your logic for negative coefficients
        #         # Example:
        #         negative_bound = self.lb[j]
        #         cond = coeff_value / negative_bound < epsilon
        #         if cond:
        #             A_modifiable[i, j] = 0
        #             # self.change_dictionary["Sparsification"]["Rows"].append(i)
        #             # self.change_dictionary["Sparsification"]["Columns"].append(j)
        #             # self.change_dictionary["Sparsification"]["Values"].append(coeff_value)
        #             # self.change_dictionary["Sparsification"]["Bounds"].append(negative_bound)
        # self.A = A_modifiable.tocsr()
                    

    def eliminate_zero_rows_operation(self, epsilon):
        original_A = self.A.copy()
        A_norm, _, _ = normalize_features(self.A, self.b)
        num_rows, num_cols = A_norm.shape
        rows_to_delete = []

        # Identify rows to be marked based on the norm criteria
        for i in range(num_rows):
            row = A_norm.getrow(i)
            row_norm = np.linalg.norm(row.toarray())
            abs_coeff = np.abs(row.data)
            
            # if row_norm / abs(self.b[i]) < epsilon:
            #     rows_to_delete.append(i)
            if np.all(abs_coeff / abs(self.b[i]) < epsilon):
                rows_to_delete.append(i)

        #print(rows_to_delete)
        rows_to_keep = []
        for i in rows_to_delete:
            if self.b[i] <= 0:
                delete = True
            else:
                rows_to_keep.append(i)
                # warnings.warn(f"Model is infeasible due to a non-redundant zero row at index {i}.")

        # Update rows_to_delete based on rows_to_keep
        rows_to_delete = [i for i in rows_to_delete if i not in rows_to_keep]

        #print(rows_to_delete)
        # Convert the matrix to a format that supports assignment, like COO or LIL
        A_modifiable = self.A.tolil()
        for i in rows_to_delete:
            A_modifiable[i, :] = 0  # Set the entire row to zero
        # Convert back to CSR format
        self.A = A_modifiable.tocsr()
        self.change_dictionary["Eliminate Zero Rows"]["Rows"] = rows_to_delete