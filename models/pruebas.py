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
from gurobipy import GRB

from opts import parse_opts
from utils_models.presolvepsilon_class import PresolvepsilonOperations
from utils_models.utils_functions import *
from utils_models.standard_model import standard_form_e1, construct_dual_model, standard_form

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
model_path_modified = os.path.join(GAMS_path_modified, 'DINAM.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_EAPP_2030_sc01_st1.mps')

model =gp.read(real_model_path)

# bounded_vars = [var for var in model.getVars() if not (var.LB == 0 and var.UB == float('inf'))]
# bounded_vars = [var for var in bounded_vars if var.LB > -GRB.INFINITY or var.UB < GRB.INFINITY]

# ifneq_constrs = [constr for constr in model.getConstrs() if constr.Sense != GRB.EQUAL]
# print(len(ifneq_constrs))    
model_standard = standard_form_e1(model)    
A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = calculate_bounds_candidates_sparse(model_standard, None, None)
lb = np.array(lb)
print(len(lb))
positive = np.where(lb >= 0)[0]
print(len(positive))