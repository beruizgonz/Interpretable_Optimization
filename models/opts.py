import argparse
import os 

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--operate_epsilon_rows', default = False, type = bool, help = 'operate the presolve epsilon rows')
    parser.add_argument('--operate_epsilon_cols', default = False, type = bool, help = 'operate the presolve epsilon cols')
    parser.add_argument('--sparsification', default = False, type = bool, help = 'operate the sparsification')
    parser.add_argument('--epsilon',default = 1e-6, type = float, help = 'the epsilon value')
    parser.add_argument('--bounds',default = False, type = bool, help = 'calculate the bounds')
    parser.add_argument('--save_path',default = 'model.json', type = str, help = 'the path to save the model')
    parser.add_argument('--read_bounds',default = False, type = bool, help = 'read the bounds')
    parser.add_argument('--change_elements',default = False, type = bool, help = 'change the element')
    parser.add_argument('--norm_abs',default = 'euclidean', type = str, help = 'normalize the absolute value')
    parser.add_argument('--local_solution', default = True, type = bool, help = 'calculate the local solution and use for the bounds')
    parser.add_argument('--sparse', default = False, type = bool, help = 'use the sparsse presolve operations')
    args = parser.parse_args()
    return args 