import json 
import os 
import gurobipy as gp
import numpy as np 

from utils_models.utils_functions import * 
from utils_models.standard_model import *
from real_objective import *

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_folder = os.path.join(project_root, 'results/prueba')
file = os.path.join(results_folder, 'epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')
model_file_new = os.path.join(project_root, 'data/real_data_new/openTEPES_EAPP_2030_sc01_st1.mps')
model_file = os.path.join(project_root, 'data/real_data/openTEPES_EAPP_2030_sc01_st1.mps')

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

A_new, _,_, _, _, _, _, _, _,_ = get_model_matrices(model)
print(A_new.shape)


# Convert the model to standard form
model_standard = standard_form_e2(model)
model_standard = set_real_objective(model_standard)
A, b, c, co, lb, ub, of_sense, cons_senses, variable_names,constraint_names = get_model_matrices(model)
normalized_model = normalize_variables(model_standard)
A_norm, b_norm, c_norm, co_norm, lb_norm, ub_norm, of_sense_norm, cons_senses_norm, variable_names_norm, constraint_names_norm = get_model_matrices(normalized_model)


col_sums_of_squares = A_norm.power(2).sum(axis=0)
column_norms = np.sqrt(np.asarray(col_sums_of_squares).ravel())
c = np.array(c)
little_cols = np.where(column_norms < 1)[0]  
print('Number of columns with norm less than 1', len(little_cols))
big_cols = np.where(column_norms > 1)[0]    

# Get the position of the variables with norm less than 1
variables = normalized_model.getVars()
variables_names = [var.VarName for var in variables]
condition = np.where(column_norms == 1)
variables_names = np.array(variables_names)
names_con = variables_names[condition]
group_dict = {}
for name in names_con:
    group = name.split('_')[0]
    if group in group_dict:
        group_dict[group].append(name)
    else:
        group_dict[group] = [name]
print(group_dict.keys())

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
