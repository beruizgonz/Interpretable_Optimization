import json
import time
import copy
import os
import gurobipy as gp
import logging
import sys
from tabulate import tabulate
from datetime import datetime
import traceback

from utils_models.presolve_class import PresolveComillas
from utils_models.sensitivity_analysis import SensitivityAnalysis
from utils_models.utils_functions import create_original_model, get_model_matrices, \
    save_json, build_model_from_json, compare_models, normalize_features, matrix_sparsification, \
    sparsification_sensitivity_analysis, build_dual_model_from_json, \
    constraint_distance_reduction_sensitivity_analysis, pre_processing_model, constraint_reduction, \
    print_model_in_mathematical_format, visual_join_sensitivity, \
    measuring_constraint_infeasibility, quality_check, sparsification_test, constraint_reduction_test, get_info_GAMS, \
    detailed_info_models, rhs_sensitivity, cost_function_sensitivity, dict2json, canonical_form, nested_dict, \
    get_primal_decisions_to_excel, store_models_matrices, get_model_in_mathematical_format

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

if __name__ == "__main__":

    # ====================================== Defining initial configuration ============================================
    config = {"get_only_GAMS_info": {"val": False,
                                     "save_xls": False},
              "get_only_GAMS_data": False,
              "skip_big_models": {"val": True,
                                  "n_constraints": 300,
                                  "n_variables": 300},
              "create_model": {"val": False,
                               "n_variables": 50000,
                               "n_constraints": 4},
              "load_model": {"val": True,
                             "load_path": 'GAMS_library',
                             "name": 'all'},
              "print_mathematical_format": False,
              "original_primal_canonical": {"val": True,
                                            "MinMode": True},
              "solve_models": True,
              "quality_check": True,
              "verbose": 0,
              "print_detail_sol": False,
              "primal_decisions2excel": False,
              "save_original_model": {"val": False,
                                      "save_name": 'transp_singleton.mps',
                                      "save_path": 'models_library'},
              "presolve": {"val": False,
                           "model_presolve": 'original_primal',
                           "print_sol": False,
                           "presolve_operations": PresolveComillas(model=None,
                                                                   perform_eliminate_zero_rows=False,
                                                                   perform_eliminate_zero_columns=False,
                                                                   perform_eliminate_singleton_equalities=False,
                                                                   perform_eliminate_kton_equalities={"val": False,
                                                                                                      "k": 5},
                                                                   perform_eliminate_singleton_inequalities=False,
                                                                   perform_eliminate_dual_singleton_inequalities=False,
                                                                   perform_eliminate_redundant_columns=False,
                                                                   perform_eliminate_implied_bounds=False,
                                                                   perform_eliminate_redundant_rows=False,
                                                                   perform_reduction_small_coefficients={"val": True,
                                                                                                         "threshold_small": 0.1},
                                                                   perform_bound_strengthening=False,
                                                                   practical_infinity=1e20
                                                                   )
                           },
              "sensitivity_analysis": {"val": True,
                                       "sensitivity_operations": SensitivityAnalysis(model=None,
                                                                                     save_path=None,
                                                                                     practical_infinity=1e20,
                                                                                     perform_reduction_small_coefficients={
                                                                                         "val": True,
                                                                                         "init_threshold": 0.0015,
                                                                                         "step_threshold": 0.2,
                                                                                         "max_threshold": 0.3}
                                                                                     )
                                       }
              }

    # ================================================== Directories to work ===========================================
    # log.info(
    #     f"{str(datetime.now())}: Preparing directories...")

    # Get the directory of the current script (main.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one levels to get to the root of the project
    project_root = os.path.dirname(current_script_dir)

    # paths to work
    data_path = os.path.join(project_root, 'data')
    current_matrices_path = os.path.join(data_path, 'current_matrices')

    # ==================================================== info GAMS LPS ===============================================
    if config['get_only_GAMS_info']['val']:
        gams_path = os.path.join(data_path, 'GAMS_library')
        get_info_GAMS(gams_path, save_excel=config['get_only_GAMS_info']['save_xls'])
        sys.exit()

    # =================================================== store GAMS data ==============================================
    if config['get_only_GAMS_data']:
        gams_path = os.path.join(data_path, 'GAMS_library')
        # store_models_matrices(gams_path)
        data = store_models_matrices(gams_path)
        sys.exit()

    # ================================= Creating or loading the original_primal model ==================================
    if config['create_model']['val'] and config['load_model']['val']:
        raise ValueError("Configuration error: 'create_model' and 'load_model' cannot both be True.")

    if config['create_model']['val']:
        log.info(
            f"{str(datetime.now())}: Creating the original_primal with {config['create_model']['n_variables']} variables "
            f"and {config['create_model']['n_constraints']} constraints...")
        original_primal_bp = create_original_model(config['create_model']['n_variables'],
                                                   config['create_model']['n_constraints'])
        model_files = [original_primal_bp]
        model_path = os.path.join(data_path, 'models_library')
        model_to_save = os.path.join(model_path, 'original_primal_bp.mps')
        original_primal_bp.write(model_to_save)

    if config['load_model']['val']:
        # log.info(
        #     f"{str(datetime.now())}: Loading models...")

        model_names = config['load_model']['name']
        model_path = os.path.join(data_path, config['load_model']['load_path'])

        if model_names == 'all':
            # Load all .mps files in the directory
            model_files = [f for f in os.listdir(model_path) if f.endswith('.mps')]
        elif isinstance(model_names, list):
            # Load specified models from the list
            model_files = [f"{name}.mps" for name in model_names if
                           os.path.exists(os.path.join(model_path, f"{name}.mps"))]
        else:
            # Single model specified
            model_files = [model_names]

    # ============================================== Iterative Process =================================================

    sparsification_results = nested_dict()
    total_models = len(model_files)

    for index, model_file in enumerate(model_files, start=1):
        
        try:
            print(model_file, '\n')
            model_to_load = os.path.join(model_path, model_file)
            model_name = model_file.replace('.mps', '')
            gp.setParam('OutputFlag', 0)
            original_primal_bp = gp.read(model_to_load)

            #log.info(f"{datetime.now()}: Processing Model {index} of {total_models} - {model_name}")

            # ========================================== Standardization of the model ==================================
            # log.info(
            #     f"{str(datetime.now())}: Standardization of the model...")
            if config["original_primal_canonical"]["val"]:
                original_primal, track_elements = canonical_form(original_primal_bp,
                                                                 minOption=config["original_primal_canonical"][
                                                                     "MinMode"])
            else:
                original_primal = pre_processing_model(original_primal_bp)

            # ===================================== saving the original_primal model in the path =======================
            if config["save_original_model"]["val"]:
                # log.info(
                #     f"{str(datetime.now())}: Saving the original_primal...")

                s_path = os.path.join(data_path, config["save_original_model"]["save_path"])
                model_to_save = os.path.join(s_path, config["save_original_model"]["save_name"])
                original_primal_bp.write(model_to_save)

            # ================================ Getting matrices and data of the original model =========================
            # log.info(
            #     f"{str(datetime.now())}: Accessing matrix A, right-hand side b, cost function c, and the bounds of "
            #     f"the original model...")
            A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)

            # ==================================== Saving matrices as json in the path =================================
            # log.info(
            #     f"{str(datetime.now())}: Saving A, b, c, lb and ub...")
            save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)

            # =========================== Creating the primal and the dual models from json files ======================
            # log.info(
            #     f"{str(datetime.now())}: Creating primal model by loading A, b, c, lb and ub...")
            created_primal = build_model_from_json(current_matrices_path)

            # log.info(
            #     f"{str(datetime.now())}: Creating dual model by loading A, b, c, lb and ub...")
            created_dual = build_dual_model_from_json(current_matrices_path)
            A_dual, b_dual, c_dual, co_dual, lb_dual, ub_dual, of_sense_dual, cons_senses_dual, variable_names = get_model_matrices(
                created_dual)

            # log.info(
            #     f"{str(datetime.now())}: Creating the normalized primal model...")
            A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
            A_norm, b_norm, A_scaler = normalize_features(A, b)
            save_json(A_norm, b_norm, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
            created_primal_norm = build_model_from_json(current_matrices_path)

            # ============================================== big models test ===========================================
            if config['skip_big_models']['val']:
                num_rows, num_cols = A.shape
                if (num_rows > config['skip_big_models']['n_constraints']) or (
                        num_cols > config['skip_big_models']['n_variables']):
                    # log.info(
                    #     f"{str(datetime.now())}: Skipping the model with {num_cols} variables and {num_rows} constraints...")
                    continue

            # ==================================== Printing model in mathematical format ===============================
            if config['print_mathematical_format']:
                log.info(
                    f"{str(datetime.now())}: Printing detailed mathematical formulations...")

                print("==================== Original Model before pre-processing ==================== ")
                print_model_in_mathematical_format(original_primal_bp)
                print("==================== Original Model after pre-processing ==================== ")
                print_model_in_mathematical_format(original_primal)
                print("==================== Model created from matrices ==================== ")
                print_model_in_mathematical_format(created_primal)
                print("==================== Normalized Model created from matrices ==================== ")
                print_model_in_mathematical_format(created_primal_norm)
                print("==================== Dual model created from matrices ==================== ")
                print_model_in_mathematical_format(created_dual)

            # ======== solving all models: original_primal_bp, original_primal, created_primal and created_dual ========

            if config['solve_models']:
                # log.info(
                #     f"{str(datetime.now())}: Solving the original_primal model (before pre-processing)...")
                original_primal_bp.setParam('OutputFlag', config['verbose'])
                try:
                    original_primal_bp.optimize()
                    original_primal_bp_sol = original_primal_bp.objVal
                except:
                    print(f"Warning: Optimal solution was not found.")
                    original_primal_bp_sol = None

                # log.info(
                #     f"{str(datetime.now())}: Solving the original_primal model (after pre-processing)...")
                original_primal.setParam('OutputFlag', config['verbose'])

                try:
                    original_primal.optimize()
                    original_primal_sol = original_primal.objVal
                except:
                    print("Warning: Optimal solution was not found.")
                    original_primal_sol = None

                # log.info(
                #     f"{str(datetime.now())}: Solving the created_primal model...")
                created_primal.setParam('OutputFlag', config['verbose'])

                try:
                    created_primal.optimize()
                    created_primal_sol = created_primal.objVal
                except:
                    print(f"Warning: Optimal solution was not found.")
                    created_primal_sol = None

                # log.info(
                #     f"{str(datetime.now())}: Solving the normalized_primal model...")
                created_primal_norm.setParam('OutputFlag', config['verbose'])

                try:
                    created_primal_norm.optimize()
                    created_primal_norm_sol = created_primal_norm.objVal
                except:
                    print(f"Warning: Optimal solution was not found.")
                    created_primal_norm_sol = None

                # log.info(
                #     f"{str(datetime.now())}: Solving the created_dual model...")
                created_dual.setParam('OutputFlag', config['verbose'])

                try:
                    created_dual.optimize()
                    created_dual_sol = created_dual.objVal
                except:
                    print(f"Warning: Optimal solution was not found.")
                    created_dual_sol = None

            # ============================ Comparing solutions: original_primal X created_primal =======================

            # printing detailed information about the models
            if config['print_detail_sol']:
                log.info(
                    f"{str(datetime.now())}: Printing detailed solution")
                detailed_info_models(original_primal_bp, original_primal, created_primal, created_primal_norm,
                                     created_dual)

            # Saving primal decision variables to excel to comparison
            if config['primal_decisions2excel']:
                log.info(
                    f"{str(datetime.now())}: Saving primal solution to excel")
                file_path = get_primal_decisions_to_excel([original_primal, created_primal, created_primal_norm])

            # ================================================ Quality check ===========================================
            if config['quality_check']:
                quality_check(original_primal_bp, original_primal, created_primal, created_primal_norm, created_dual,
                              tolerance=1e-2)
                # log.info(
                #     f"{str(datetime.now())}: Quality check passed...")

            # ============================================= Presolve operations ========================================
            if config['presolve']['val']:
                if config['presolve']['model_presolve'] == 'original_primal':
                    # log.info(
                    #     f"{str(datetime.now())}: Presolve operations with the original_primal...")
                    model_to_use = original_primal
                else:
                    log.info(
                        f"{str(datetime.now())}: Presolve operations with the created_dual...")
                    model_to_use = created_dual

                presolve_instance = config['presolve']['presolve_operations']
                presolve_instance.model = model_to_use

                A, b, c, lb, ub, of_sense, cons_senses, co, variable_names, changes_dictionary, operation_table = (
                    presolve_instance.orchestrator_presolve_operations())

                log.info(
                    f"{str(datetime.now())}:\n{tabulate(operation_table, headers=['Operation', 'Rows', 'Columns', 'Non-Zeros'], tablefmt='grid')}"
                )

                if config['presolve']['print_sol']:
                    print("===================== Original model =====================")
                    print_model_in_mathematical_format(original_primal_bp)

                    print("===================== Model before presolve =====================")
                    print_model_in_mathematical_format(model_to_use)

                    save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
                    model_after = build_model_from_json(current_matrices_path)

                    print("===================== Model after presolve =====================")
                    print_model_in_mathematical_format(model_after)

            # ============================================ Sensitivity Analysis ========================================

            if config['sensitivity_analysis']['val']:
                sa_instance = config['sensitivity_analysis']['sensitivity_operations']
                model_to_use_primal = original_primal
                model_to_use_dual = created_dual
                sa_instance.save_path = current_matrices_path

                # For the primal model
                log.info(
                    f"{str(datetime.now())}:Sensitivity Analysis with primal model..."
                )
                sa_instance_primal = copy.deepcopy(sa_instance)
                sa_instance_primal.model = model_to_use_primal

                start_time = time.time()
                sparsification_results[model_name]['primal'] = sa_instance_primal.orchestrator_sensitivity_operations()
                sparsification_results[model_name]['primal']['time_required'] = time.time() - start_time
                sparsification_results[model_name]['primal']['mathematical_model'] = get_model_in_mathematical_format(
                    model_to_use_primal)

                # For the dual model
                log.info(
                    f"{str(datetime.now())}:Sensitivity Analysis with dual model..."
                )
                sa_instance_dual = copy.deepcopy(sa_instance)
                sa_instance_dual.model = model_to_use_dual

                start_time = time.time()
                sparsification_results[model_name]['dual'] = sa_instance_dual.orchestrator_sensitivity_operations()
                sparsification_results[model_name]['dual']['time_required'] = time.time() - start_time
                sparsification_results[model_name]['dual']['mathematical_model'] = get_model_in_mathematical_format(
                    model_to_use_dual)

            # Saving the dictionary
            dict2json(sparsification_results, 'sparsification_results1.json')

        except Exception as e:
            print(f"general error found:\n{traceback.format_exc()}\n{e}")
