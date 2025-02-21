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
from utils_models.standard_model import standard_form_e2
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
real_model_path = os.path.join(real_data_path,  'openTEPES_9n_2030_sc01_st1.mps')

#real_model =gp.read(real_model_path)
model_gams = gp.read(model_path_modified)

# bounded_vars = [var for var in model.getVars() if not (var.LB == 0 and var.UB == float('inf'))]
# bounded_vars = [var for var in bounded_vars if var.LB > -GRB.INFINITY or var.UB < GRB.INFINITY]

# ifneq_constrs = [constr for constr in model.getConstrs() if constr.Sense != GRB.EQUAL]
# print(len(ifneq_constrs))    

def toy_model():
    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVar(vtype=GRB.CONTINUOUS, lb = 0,  name="x")
    y = m.addVar(vtype=GRB.CONTINUOUS, lb = 0, name="y")
    # Set objective
    m.setObjective(x + y, GRB.MINIMIZE)
    # Add constraint: x + 2 y <= 1
    m.addConstr(-2*x + y <= 10, "c0")
    m.addConstr(0.2*x + y <= 20, "c1")
    m.addConstr(-x -y <= -5, "c2")
    # Return the model
    m.update()
    return m

model = toy_model()
model_standard = standard_form_e2(model_gams)    
model_standard.setParam('OutputFlag', 0)
model_standard.optimize()
print(model_standard.objVal)
A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = calculate_bounds_candidates_sparse_improve(model_standard, None, None)
# print the number of ub that are not inf
ub_not_inf = [u for u in ub if u != float('inf')]
print(len(ub_not_inf))
print(len(ub))
print(len(ub) - len(ub_not_inf))    
# Create the model 
model_bounds = build_model(A, b, c, co, lb, ub, of_sense, cons_senses, variable_names)
model_bounds.setParam('OutputFlag', 0)  
model_bounds.optimize()
print(model_bounds.objVal)  