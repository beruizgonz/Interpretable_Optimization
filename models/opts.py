import argparse
import os 

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--operate_epsilon_rows', default = True, type = bool, help = 'operate the presolve epsilon rows')
    parser.add_argument('--operate_epsilon_cols', default = False, type = bool, help = 'operate the presolve epsilon cols')
    parser.add_argument('--epsilon',default = 1e-6, type = float, help = 'the epsilon value')
    parser.add_argument('--save_path',default = 'model.json', type = str, help = 'the path to save the model')
    args = parser.parse_args()
    return args 