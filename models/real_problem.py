import json
import time
import os
import gurobipy as gp
import logging
import sys
from tabulate import tabulate
from datetime import datetime
import pickle
import math
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import gc

from opts import parse_opts
from utils_models.presolvepsilon_class import PresolvepsilonOperations
from utils_models.utils_functions import *
from utils_models.standard_model import *

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

# Load the data information
opts = parse_opts()
#opts.eliminate_zero_rows_epsilon = True


#The path to the data
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'DINAM.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'PDI.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_EAPP_2030_sc01_st1.lp')
trace_file = os.path.join(real_data_path)
# Coverting the model to standard form
print(os.path.exists(model_path_modified))
model = gp.read(real_model_path)
# model.optimize()
# print('Model optimized: ', model.objVal)
# Geth the variables and constraints
# Get the number of variables and constraints

#model_canonical = canonical_form(model)
# model_standard = standard_form(model)
# print('Model standard form')
# model_standard.setParam('OutputFlag', 0)
# model_standard.optimize()
# print('Model standard optimized: ', model_standard.objVal)
model_standard = standard_form_e1(model)
model_standard.setParam('OutputFlag', 0)
model_standard.optimize()
print('Model standard optimized: ', model_standard.objVal)
# A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model_standar1)

# variables = model_standard.getVars()
# constraints = model_standard.getConstrs()
# # Verify that all constraints are in the standard form ==
# sense = [c.Sense for c in constraints]

# lower_bound = [var.LB for var in variables]
# upper_bound = [var.UB for var in variables]
# # convert upper bound to a numpy array
# upper_bound = np.array(upper_bound)
# lower_bound = np.array(lower_bound)
# print('Upper bounds: ', np.all(upper_bound == float('inf'))) 
# print('Lower bounds: ', np.all(lower_bound == 0))

# print('Model converted to standard form')
# # model_standard.optimize()
# # # # print('Model standard optimized: ', model_standard.objVal)
#lb,ub = calculate_bounds(model_standar1)
# lb1, ub1 = calculate_bounds(model_standard)
# #¡print(len(lb), len(ub))ç
# print(len(lb1), len(ub1))
# print(ub1)
      

# # # print('Lower bounds: ', lb)
# # # print('Upper bounds: ', ub)
# # # #min_activity = min_activity.data
#A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model_standard)
A_sparse, b, c, co, lb_new, ub_new, of_sense, cons_sense, variable_names  = calculate_bounds_candidates(model_standard, trace_file, max_iterations=20)
# print(len(lb_new), len(ub_new))
# # #lb_new, ub_new = calculate_bounds_candidates(model_standard)
#model_bounds = build_model_from_json('model_matrices.json')   
# print_model_in_mathematical_format(model_bounds) 
model_bounds =build_model(A_sparse, b, c, co, lb_new, ub_new, of_sense, cons_sense, variable_names)
model_bounds.setParam('OutputFlag', 0)
model_bounds.optimize()
print('Model bounds optimized: ', model_bounds.objVal)
# # # print('Lower bounds: ', lb)
# print('Upper bounds: ', ub)
# print('Min activity: ', min_activity)
# print('Min activity scipy: ', min_activity1.shape)


# model_bounds =build_model(A, b, c, co, lb, ub, of_sense, cons_senses, variable_names)
# model_bounds.setParam('OutputFlag', 0)
# model_bounds.optimize()
# print('Model bounds optimized: ', model_bounds.objVal)
# opts.change_elements = True
# presolve  = PresolvepsilonOperations(model_standard, eliminate_zero_rows_epsilon= False, opts=opts)
# presolve.orchestrator_presolve_operations(model_standard)


