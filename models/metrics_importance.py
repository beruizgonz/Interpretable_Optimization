import os
import gurobipy as gp
import numpy as np
from scipy.stats import rankdata
import scipy.sparse as sp   

from description_data.description_open_TEPES import *
from description_data.marginal_values_open_TEPES import *
from description_data.plot_matrix import *
from utils_models.utils_functions import *
from utils_models.standard_model import *
from example_models import *



# PATH TO THE DATA
real_data_path = os.path.join(parent_path, 'data/real_data')
open_tepes_9n = os.path.join(real_data_path, 'openTEPES_EAPP_2030_sc01_st1.mps')
gams_data_path = os.path.join(parent_path, 'data/GAMS_library_modified')
gams_model = os.path.join(gams_data_path, 'ROBERT.mps')

# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/marginal_values/real_problems')
results_simplified_constraints = os.path.join(results_folder, 'simplified_constraints_epsilon/importance_openTEPES_EAPP_2030_sc01_st1_constraints.json')
results_simplified_variables = os.path.join(results_folder, 'simplified_variables_percentage/openTEPES_EAPP_2030_sc01_st1_variables.json')

# Save paths 
interactive_figures = os.path.join(parent_path, 'figures_new/interactive_figures/marginal_value/simplified_constraints_percentage')
figures = os.path.join(parent_path, 'figures_new')
real_problems = os.path.join(interactive_figures, 'real_problems')
save_metrics_importance = os.path.join(figures, 'metrics_importance')
metrics_importance_real_problems = os.path.join(save_metrics_importance, 'real_problems')
save_path_metrics_pre = os.path.join(metrics_importance_real_problems, 'pre_solve')
# In this code it is importance to notice that the model we are going to use is normalized. All variables are normalized to the same scale. LB = 0, UB = 1

def get_normalize_model(model_path):
    model = gp.read(model_path)  
    model_standard = standard_form_e2(model)
    model_standard = set_real_objective(model_standard)
    model_normalize = normalize_variables(model_standard)
    return model_normalize

def importance_constraints_norm(model): 
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    b_array = np.array(b)
    b_array = b_array.reshape(-1, 1)
    c_array = np.array(c)
    sum_row = np.sum(np.abs(A), axis = 1)
    imp_constr = b_array- sum_row
    return imp_constr

def name_constraint_importance(imp_constr, model): 
    constraint_names = model.getConstrs()
    constraint_names = [constr.ConstrName for constr in constraint_names]
    imp_constr = np.array(imp_constr)
    imp_constr = imp_constr.reshape(-1, 1)
    imp_constr = np.abs(imp_constr)
    imp_constr = imp_constr/np.sum(imp_constr)
    imp_constr = imp_constr.tolist()
    imp_constr = [imp_constr[i][0] for i in range(len(imp_constr))]
    imp_constr_dict = dict(zip(constraint_names, imp_constr))
    names, asos_elements, group_dict, invert_dict = groups_by_constraints(model)
    imp_constr_group = dict.fromkeys(group_dict.keys(), 0)
    for key, value in imp_constr_dict.items():
        name_group = invert_dict[key]
        imp_constr_group[name_group] += value

    # divided by the number of elements in the group
    for key, value in imp_constr_group.items():
        imp_constr_group[key] = imp_constr_group[key]/len(group_dict[key   ])
    imp_constr_group = dict(sorted(imp_constr_group.items(), key=lambda item: item[1], reverse = True))
    return imp_constr_group, group_dict

def importance_variable_of(model): 
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    c_array = np.array(c)
    sum_c = np.sum(np.abs(c_array))
    imp_var = np.abs(c_array)/sum_c
    return imp_var

def importance_variable_of_group(imp_var, model): 
    names, asos_elements, group_dict, invert_dict = groups_by_variables(model)
    A,b,c,co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    imp_var = np.array(imp_var)
    imp_var = imp_var.reshape(-1, 1)
    imp_var = imp_var.tolist()
    imp_var = [imp_var[i][0] for i in range(len(imp_var))]
    imp_var_dict = dict(zip(variable_names, imp_var))
    imp_var_group = dict.fromkeys(group_dict.keys(), 0)
    for key, value in imp_var_dict.items():
        name_group = invert_dict[key]
        imp_var_group[name_group] += value
    imp_var_group = dict(sorted(imp_var_group.items(), key=lambda item: item[1], reverse = True))
    return imp_var_group, group_dict

def importance_variable_constraints(model):
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    
    A_abs = abs(A)  # This works efficiently for sparse matrices

    # Compute row-wise sum (sum along axis 1)
    row_sums = np.array(A_abs.sum(axis=1)).flatten()  # Convert sparse sum result to 1D NumPy array

    # Avoid division by zero for rows that sum to 0
    row_sums[row_sums == 0] = 1  

    # Normalize each row
    imp_var_const = A_abs.multiply(1 / row_sums[:, np.newaxis])  # Element-wise division
    impp_var_cost = csr_matrix(imp_var_const)
    return impp_var_cost  # Returns a sparse matrix

def importance_constraint_for_variable(model):
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    imp_var_constr = importance_variable_constraints(model)
    rows_cols = np.array(imp_var_constr.sum(axis= 0)).flatten()
    imp_var_constr_norm = imp_var_constr.multiply(1/rows_cols)
    imp_var_constr_norm = csr_matrix(imp_var_constr_norm)
    print(imp_var_constr_norm.shape)
    return imp_var_constr_norm

def importance_constr_for_var_group(imp_var_constr, model):
    names_constr, asos_constr, group_constraints, invert_constr = groups_by_constraints(model)
    names_var, asos_var, group_variables, invert_var = groups_by_variables(model)
    association_dict = {(group_constr, group_var): 0 for group_constr in group_constraints for group_var in group_variables}
    pos_names = map_position_to_names(model)
    for position, name in pos_names.items():
        group_constr = invert_constr[name[0]]
        group_var = invert_var[name[1]]
        association_dict[(group_constr, group_var)] += imp_var_constr[position]
    # Normalize the values by group of constraints  
    for key, value in association_dict.items():
        group_constr = key[1]
        association_dict[key] = association_dict[key]/len(group_variables[group_constr])
    association_dict = dict(sorted(association_dict.items(), key=lambda item: item[1], reverse = True))
    return association_dict, group_constraints, group_variables


def importance_variable_constraints_group(imp_var_const, model):
    names_constr, asos_constr, group_constraints, invert_constr = groups_by_constraints(model)
    names_var, asos_var, group_variables, invert_var = groups_by_variables(model)
    association_dict = {(group_constr, group_var): 0 for group_constr in group_constraints for group_var in group_variables}
    pos_names = map_position_to_names(model)
    for position, name in pos_names.items():
        group_constr = invert_constr[name[0]]
        group_var = invert_var[name[1]]
        association_dict[(group_constr, group_var)] += imp_var_const[position]
    # Normalize the values by group of constraints  
    for key, value in association_dict.items():
        group_constr = key[0]
        association_dict[key] = association_dict[key]/len(group_constraints[group_constr])
    association_dict = dict(sorted(association_dict.items(), key=lambda item: item[1], reverse = True))
    return association_dict, group_constraints, group_variables

def synergy_variables(model): 
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    # Create a matrix with shape (n, n) where n is the number of variables
   # Get number of columns (variables)
    A_sparse = csr_matrix(A)    
    n = A_sparse.shape[1]

    # Extract the first row (efficiently for sparse matrices)
    A_first_row = A_sparse.getrow(0).toarray().flatten()  # Convert to dense array

    # Initialize synergy array (dense, will be converted back to sparse)
    synergy_variable_constraints = np.zeros((1, n))

    # Get the first element of the row
    A_00 = A_first_row[16]

    # Compute synergy values, avoiding division by zero
    for i in range(n):
        if A_first_row[i] != 0:  # Only compute if variable exists
            synergy_variable_constraints[0, i] = -A_00 / A_first_row[i]
    print(synergy_variable_constraints)


class RedundantConstraints:
    def __init__ (self, model = None, opts =None):
        self.model = model
        self.opts = opts
        self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names, self.constraint_names = get_model_matrices(self.model)
        self.A_sparse = csr_matrix(self.A)
        self.num_rows = self.A_sparse.shape[0]
        self.num_cols = self.A_sparse.shape[1]

        self.c = np.array(self.c)   

        self.opts = opts
    
    def orchestator_metrics_constraints(self, model, epsilon):
        if self.opts.angles:
            self.cosine_constraints(model)
        elif self.opts.coefficientb_A:
            self.coefficientb_A_constraints(model)
        elif self.opts.coefficientc_A:
            self.coefficientc_A_constraints(model)
        elif self.opts.PRMAC:
            self.PRMAC_constraints(model)
    
    def ranking_vector(self,metric, order ='descending'):
        """
        Rank the constraints according to the metric
        """
        if order == 'descending':
            ranking = np.argsort(metric)[::-1]
        else:
            ranking = np.argsort(metric)
        return ranking
    
    def ranking_matrix(self,metric, order ='descending'):
        """
        For each variable rank the constraints according to the metric
        """
        # find the duplicated values for each column
        R = rankdata(metric, method = 'average')

        return R
    
    def rankdata_sparse(self, matrix, decimal_places=8):
        """
        Applies rankdata on a sparse matrix column-wise without converting the whole matrix to dense.
        
        Parameters:
        - matrix: scipy sparse matrix (CSR or CSC format; CSR is converted to CSC internally)
        - decimal_places: Number of decimal places to round to (to handle precision issues)
        
        Returns:
        - A sparse CSR matrix with ranked values for the nonzero entries.
        
        The function uses the CSC representation to quickly access nonzero entries of each column.
        For each column, it:
        1. Retrieves the nonzero row indices and values.
        2. Rounds the nonzero values.
        3. Computes their ranks (using, e.g., scipy.stats.rankdata).
        4. Collects the row indices, column indices, and computed ranks.
        Finally, it constructs a new CSR matrix from these arrays.
        """

        if not sp.issparse(matrix):
            raise ValueError("Input must be a sparse matrix.")

        # Convert to CSC format for efficient column-wise operations.
        matrix = matrix.tocsc()
        n_rows, n_cols = matrix.shape

        # Lists to collect data for the ranked matrix.
        row_indices_list = []
        col_indices_list = []
        data_list = []

        # Iterate over each column using CSC structure.
        for col in range(n_cols):
            start = matrix.indptr[col]
            end = matrix.indptr[col + 1]
            if start < end:
                # Get the row indices and nonzero values for this column.
                col_rows = matrix.indices[start:end]
                col_values = matrix.data[start:end]
                
                # Round the nonzero values to avoid precision issues.
                rounded_values = np.round(col_values, decimals=decimal_places)
                
                # Compute ranks on the rounded nonzero values.
                # Note: rankdata returns ranks in the order of the array.
                col_ranks = rankdata(rounded_values)
                
                # Store the results.
                row_indices_list.append(col_rows)
                col_indices_list.append(np.full(col_ranks.shape, col, dtype=int))
                data_list.append(col_ranks)

        if row_indices_list:
            # Concatenate the results from all columns.
            all_rows = np.concatenate(row_indices_list)
            all_cols = np.concatenate(col_indices_list)
            all_data = np.concatenate(data_list)
        else:
            # No nonzero entries? Return an empty matrix of the same shape.
            return csr_matrix(matrix.shape, dtype=float)

        # Build the ranked matrix as a CSR matrix.
        ranked_matrix = csr_matrix((all_data, (all_rows, all_cols)), shape=(n_rows, n_cols))
        
        return ranked_matrix
# Convert back to efficient CSR format


    def find_column_duplicates(self, matrix):
        matrix = np.array(matrix, dtype=float)
        matrix = np.round(matrix, 5)
        for i, col in enumerate(matrix.T):  # Transpose to iterate over columns
            unique_vals, counts = np.unique(col, return_counts=True)
            duplicates = unique_vals[counts > 1]
            print(f"Column {i+1}:")
            print("  Repeated Values:", duplicates)
            print("  Counts:", counts[counts > 1])


    def cosine_constraints(self):
        """
        Compute the angles between the constraints
        """
        rows_norm = np.sqrt(self.A_sparse.multiply(self.A_sparse).sum(axis=1))
        c_norm = np.linalg.norm(self.c)
        c_vector = self.c.reshape(1,-1)
        angles_metric = self.A_sparse.multiply(c_vector) / (rows_norm * c_norm)
        constraint_angles = np.sum(angles_metric, axis = 1)
        constraint_angles = constraint_angles.flatten()
        ranking = self.find_column_duplicates(constraint_angles)
        print(ranking)
        return angles_metric
    
    def PRMAC_constraints(self):
        """
        Compute the PRMAC metric for the constraints. 
        It is based on the paper: A class of algorithms for solving LP problems by prioritizing the constraints
        """
        # PRMAC_matrix = np.zeros((self.num_rows, self.num_cols))
        # for i, row in enumerate(self.A_sparse):
        self.c = np.array(self.c).flatten()
        
        # Copy the sparse matrix to avoid modifying the original.
        PRMAC_matrix = self.A_sparse.copy()
        
        # For each nonzero element, subtract the corresponding objective value (using the column index)
        # and take the absolute value.
        # self.A_sparse.indices contains the column indices for each nonzero element.
        PRMAC_matrix.data = np.abs(PRMAC_matrix.data - self.c[PRMAC_matrix.indices])
        # Compute ranking on the resulting sparse matrix.
        ranking = self.rankdata_sparse(PRMAC_matrix)
        print(ranking)
        
        
    
    
    #def utility_matrix(self):


if __name__ == '__main__':
    example1 = numerical_example()
    model = gp.read(open_tepes_9n)
    redundant_constraints = RedundantConstraints(example1)
    redundant_constraints.PRMAC_constraints()
