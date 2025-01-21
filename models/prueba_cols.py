import json 
import os 
import gurobipy as gp
import numpy as np 

from utils_models.utils_functions import * 
from utils_models.standard_model import *

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_folder = os.path.join(project_root, 'results/prueba')
file = os.path.join(results_folder, 'epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')
model_file_new = os.path.join(project_root, 'data/real_data_new/openTEPES_EAPP_2030_sc01_st1.mps')
model_file = os.path.join(project_root, 'data/real_data/openTEPES_9n_2030_sc01_st1.mps')

GAMS_path = os.path.join(project_root, 'data/GAMS_library')    
real_data_path = os.path.join(project_root, 'data/real_data_new')
# with open(file, 'r') as f:
#     data = json.load(f)

# data_name  = 'openTEPES_EAPP_2030_sc01_st1'
# get_data = data[data_name]
# model_primal = get_data['primal']
# obj = model_primal['objective_function']
# zero_cols = model_primal['columns_changed']


# # Calculate the euclidean norm of the columns 
# print(os.path.exists(model_file))
model = gp.read(model_file)
# model.setParam('OutputFlag', 0)
# model.optimize()
# print(model.ObjVal)




# Convert the model to standard form
model_standard = standard_form_e1(model)
A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)
A_norm, _, _ = normalize_features_sparse(A,b)

col_sums_of_squares = A_norm.power(2).sum(axis=0)
column_norms = np.sqrt(np.asarray(col_sums_of_squares).ravel())
c = np.array(c)
little_cols = np.where(column_norms < 1)[0]  
print(len(little_cols))
# any value of z that is different from 0 or 1
non_zero_c = np.where((c != 1) & (c != 0))[0]
print(len(non_zero_c))
# How many coefficients are zero in the objective function
# def euclidean_norm_columns(A):
#     """
#     Compute the euclidean norm of the columns of a matrix A
#     """
#     return np.linalg.norm(A, axis = 0)

# # Compute the euclidean norm of the columns
# norm = euclidean_norm_columns(A)
# print(norm)
# From model get the objective function
# obj = model.getObjective()
# print(obj)

# for file in os.listdir(GAMS_path):
#     if  file.endswith('.mps') and not file.endswith('ORANI.mps'):
#         model_path = os.path.join(GAMS_path, file)
#         print(model_path)
#         model = gp.read(model_path)
#         model.setParam('OutputFlag', 0)
#         model.optimize()
#         print(model.ObjVal)
#         model_path_modified = os.path.join(real_data_path, file)
#         model_modified = gp.read(model_path_modified)
#         model_modified.setParam('OutputFlag', 0)
#         model_modified.optimize()
#         print(model_modified.ObjVal)
