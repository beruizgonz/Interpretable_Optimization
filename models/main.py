import json
import os
import gurobipy as gp
import logging
import sys

from datetime import datetime

from Interpretable_Optimization.models.utils_models.presolve_class import PresolveComillas
from Interpretable_Optimization.models.utils_models.utils_functions import create_original_model, get_model_matrices, \
    save_json, build_model_from_json, compare_models, normalize_features, matrix_sparsification, \
    sparsification_sensitivity_analysis, build_dual_model_from_json, \
    constraint_distance_reduction_sensitivity_analysis, pre_processing_model, constraint_reduction, \
    print_model_in_mathematical_format, visual_join_sensitivity, \
    measuring_constraint_infeasibility, quality_check, sparsification_test, constraint_reduction_test, get_info_GAMS, \
    detailed_info_models, rhs_sensitivity, cost_function_sensitivity, dict2json, canonical_form, nested_dict

from Interpretable_Optimization.models.utils_models.utils_presolve import get_row_activities, \
    eliminate_implied_bounds, small_coefficient_reduction, eliminate_zero_columns, \
    eliminate_singleton_equalities, eliminate_zero_rows, eliminate_doubleton_equalities, eliminate_kton_equalities, \
    eliminate_singleton_inequalities, eliminate_dual_singleton_inequalities, eliminate_redundant_columns, \
    eliminate_redundant_rows

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

if __name__ == "__main__":

    # ====================================== Defining initial configuration ============================================
    config = {"get_only_GAMS_info": {"val": False,
                                     "save_xls": False},
              "create_model": {"val": False,
                               "n_variables": 50000,
                               "n_constraints": 4},
              "load_model": {"val": True,
                             "load_path": 'presolve',
                             "name": 'transp_singleton'},
              "print_mathematical_format": False,
              "original_primal_canonical": True,
              "solve_models": False,
              "quality_check": False,
              "verbose": 0,
              "print_detail_sol": False,
              "save_original_model": {"val": False,
                                      "save_name": 'transp_singleton.mps',
                                      "save_path": 'models_library'},
              "rhs_sensitivity": False,
              "cost_sensitivity": False,
              "presolve_operations": {"eliminate_zero_rows": False,
                                      "eliminate_zero_columns": False,
                                      "eliminate_singleton_equalities": False,
                                      "eliminate_doubleton_equalities": False,
                                      "eliminate_kton_equalities": False,
                                      "eliminate_singleton_inequalities": False,
                                      "eliminate_dual_singleton_inequalities": False,
                                      "eliminate_redundant_columns": False,
                                      "eliminate_redundant_rows": False,
                                      "eliminate_implied_bounds": True},
              "test_sparsification": {"val": False,
                                      "threshold": 0.13},
              "test_constraint_red": {"val": False,
                                      "threshold": 0.8},
              "sparsification_sa": {"val": False,
                                    "plots": False,
                                    "prints": False,
                                    "max_threshold": 0.3,
                                    "init_threshold": 0.01,
                                    "step_threshold": 0.001},
              "euclidian_reduction_sensitive_analysis": {"val": False,
                                                         "max_threshold": 1.4,
                                                         "init_threshold": 0.1,
                                                         "step_threshold": 0.01},
              "create_presolve": False
              }

    # ================================================== Directories to work ===========================================
    log.info(
        f"{str(datetime.now())}: Preparing directories...")

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
        log.info(
            f"{str(datetime.now())}: Loading models...")

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
        model_to_load = os.path.join(model_path, model_file)
        model_name = model_file.replace('.mps', '')
        gp.setParam('OutputFlag', 0)
        original_primal_bp = gp.read(model_to_load)

        log.info(f"{datetime.now()}: Processing Model {index} of {total_models} - {model_name}")

        # ============================================ Standardization of the model ====================================
        log.info(
            f"{str(datetime.now())}: Standardization of the model...")
        if config["original_primal_canonical"]:
            original_primal, track_elements = canonical_form(original_primal_bp, minOption=True)
        else:
            original_primal = pre_processing_model(original_primal_bp)

        # ======================================= saving the original_primal model in the path =========================
        if config["save_original_model"]["val"]:
            log.info(
                f"{str(datetime.now())}: Saving the original_primal...")

            s_path = os.path.join(data_path, config["save_original_model"]["save_path"])
            model_to_save = os.path.join(s_path, config["save_original_model"]["save_name"])
            original_primal_bp.write(model_to_save)

        # ================================= Getting matrices and data of the original model ============================
        log.info(
            f"{str(datetime.now())}: Accessing matrix A, right-hand side b, cost function c, and the bounds of "
            f"the original model...")
        A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)

        # ====================================== Saving matrices as json in the path ===================================
        log.info(
            f"{str(datetime.now())}: Saving A, b, c, lb and ub...")
        save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)

        # ============================= Creating the primal and the dual models from json files ========================
        log.info(
            f"{str(datetime.now())}: Creating primal model by loading A, b, c, lb and ub...")
        created_primal = build_model_from_json(current_matrices_path)

        log.info(
            f"{str(datetime.now())}: Creating dual model by loading A, b, c, lb and ub...")
        created_dual = build_dual_model_from_json(current_matrices_path)
        A_dual, b_dual, c_dual, co_dual, lb_dual, ub_dual, of_sense_dual, cons_senses_dual, variable_names = get_model_matrices(
            created_dual)

        # ====================================== Printing model in mathematical format =================================
        if config['print_mathematical_format']:
            log.info(
                f"{str(datetime.now())}: Printing detailed mathematical formulations...")

            print("==================== Original Model before pre-processing ==================== ")
            print_model_in_mathematical_format(original_primal_bp)
            print("==================== Original Model after pre-processing ==================== ")
            print_model_in_mathematical_format(original_primal)
            print("==================== Model created from matrices ==================== ")
            print_model_in_mathematical_format(created_primal)
            print("==================== Dual model created from matrices ==================== ")
            print_model_in_mathematical_format(created_dual)

        # ========== solving all models: original_primal_bp, original_primal, created_primal and created_dual ==========
        if config['solve_models']:
            log.info(
                f"{str(datetime.now())}: Solving the original_primal model (before pre-processing)...")
            original_primal_bp.setParam('OutputFlag', config['verbose'])
            try:
                original_primal_bp.optimize()
                original_primal_bp_sol = original_primal_bp.objVal
            except:
                print(f"Warning: Optimal solution was not found.")
                original_primal_bp_sol = None

            log.info(
                f"{str(datetime.now())}: Solving the original_primal model (after pre-processing)...")
            original_primal.setParam('OutputFlag', config['verbose'])

            try:
                original_primal.optimize()
                original_primal_sol = original_primal.objVal
            except:
                print("Warning: Optimal solution was not found.")
                original_primal_sol = None

            log.info(
                f"{str(datetime.now())}: Solving the created_primal model...")
            created_primal.setParam('OutputFlag', config['verbose'])

            try:
                created_primal.optimize()
                created_primal_sol = created_primal.objVal
            except:
                print(f"Warning: Optimal solution was not found.")
                created_primal_sol = None

            log.info(
                f"{str(datetime.now())}: Solving the created_dual model...")
            created_dual.setParam('OutputFlag', config['verbose'])

            try:
                created_dual.optimize()
                created_dual_sol = created_dual.objVal
            except:
                print(f"Warning: Optimal solution was not found.")
                created_dual_sol = None

        # ============================== Comparing solutions: original_primal X created_primal =========================

        # printing detailed information about the models
        if config['print_detail_sol']:
            detailed_info_models(original_primal_bp, original_primal, created_primal, created_dual)

        # =================================================  Normalization =============================================
        log.info(
            f"{str(datetime.now())}: Solving the normalized primal model...")

        A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(original_primal)
        A_norm, A_scaler = normalize_features(A)
        b_norm = b / A_scaler
        save_json(A_norm, b_norm, c, lb, ub, of_sense, cons_senses, current_matrices_path)
        created_primal_norm = build_model_from_json(current_matrices_path)
        created_primal_norm.setParam('OutputFlag', config['verbose'])
        created_primal_norm.optimize()

        # ================================================== Quality check =============================================
        if config['quality_check']:
            quality_check(original_primal_bp, original_primal, created_primal, created_dual, tolerance=1e-2)
            log.info(
                f"{str(datetime.now())}: Quality check passed...")

        # =============================================== Presolve operations ==========================================
        current_model = original_primal.copy()
        print_model_in_mathematical_format(current_model)
        presolve_instance = PresolveComillas(model=current_model,
                                             perform_eliminate_zero_rows=False,
                                             perform_eliminate_zero_columns=False,
                                             perform_eliminate_singleton_equalities=False,
                                             perform_eliminate_kton_equalities=False,
                                             k=2,
                                             perform_eliminate_singleton_inequalities=True)

        A, b, c, lb, ub, of_sense, cons_senses, co, variable_names, changes_dictionary, operation_table = (
            presolve_instance.orchestrator_presolve_operations())

        save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path, co, variable_names)
        current_model = build_model_from_json(current_matrices_path)
        current_model.update()
        print_model_in_mathematical_format(current_model)

        if config['presolve_operations']['eliminate_zero_rows']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate zero rows")
            print_model_in_mathematical_format(current_model)
            current_model, feedback_zero_rows = eliminate_zero_rows(current_model, current_matrices_path)

        if config['presolve_operations']['eliminate_zero_columns']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate zero columns")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, feedback_zero_columns = eliminate_zero_columns(current_model, current_matrices_path)

        if config['presolve_operations']['eliminate_singleton_equalities']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate singleton equalities")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, solution_singleton_equalities = eliminate_singleton_equalities(current_model,
                                                                                          current_matrices_path)

        if config['presolve_operations']['eliminate_doubleton_equalities']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate doubleton equalities")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model = eliminate_doubleton_equalities(current_model, current_matrices_path)

        if config['presolve_operations']['eliminate_kton_equalities']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate kton equalities")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, kton_dict = eliminate_kton_equalities(current_model, current_matrices_path, 3)

        if config['presolve_operations']['eliminate_singleton_inequalities']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate singleton inequalities")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, feedback_constraint_single_in, feedback_variable_single_in = eliminate_singleton_inequalities(
                current_model, current_matrices_path)

        if config['presolve_operations']['eliminate_dual_singleton_inequalities']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate dual singleton inequalities")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, feedback_constraint_single_in, feedback_variable_single_in = (
                eliminate_dual_singleton_inequalities(current_model, current_matrices_path))

        if config['presolve_operations']['eliminate_redundant_columns']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate redundant columns")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, feedback_constraint_red_col, feedback_variable_red_col = (
                eliminate_redundant_columns(current_model, current_matrices_path))

        if config['presolve_operations']['eliminate_redundant_rows']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate redundant rows")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, feedback_constraint = eliminate_redundant_rows(current_model, current_matrices_path)

        if config['presolve_operations']['eliminate_implied_bounds']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - eliminate implied bounds")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, feedback_constraint = eliminate_implied_bounds(current_model, current_matrices_path)

        if config['presolve_operations']['small_coefficient_reduction']:
            log.info(
                f"{str(datetime.now())}: Presolve operations - small coefficient reduction")
            current_model.update()
            print_model_in_mathematical_format(current_model)
            current_model, changes = small_coefficient_reduction(current_model)

        # ============================================ rhs sensitivity analysis ========================================
        if config['rhs_sensitivity']:
            log.info(
                f"{str(datetime.now())}: Right-hand-side sensitivity analysis:")
            rhs_dec, rhs_inc = rhs_sensitivity(original_primal_bp)

        # ======================================== cost vector sensitivity analysis ====================================
        if config['cost_sensitivity']:
            log.info(
                f"{str(datetime.now())}: Cost vector sensitivity analysis:")
            cv_dec, cv_inc = cost_function_sensitivity(original_primal_bp)

        # ================================================ Test Sparsification =========================================
        if config['test_sparsification']['val']:
            log.info(
                f"{str(datetime.now())}: Results of the sparsification test:")
            sparsification_test(original_primal, config, current_matrices_path)

        # ============================================ Test constraint reduction =======================================
        if config['test_constraint_red']['val']:
            log.info(
                f"{str(datetime.now())}: Results of the constraint reduction test:")
            constraint_reduction_test(original_primal, config, current_matrices_path)

        # =================== Sensitivity analysis on matrix sparsification for different thresholds ===================
        if config['sparsification_sa']['val']:
            log.info(
                f"{str(datetime.now())}: Sensitivity analysis on matrix sparsification for different thresholds:")

            log.info(f"{datetime.now()}: Processing primal...")
            eps_p, of_p, dv_p, ind_p, cviol_p, of_dec_p = sparsification_sensitivity_analysis(current_matrices_path,
                                                                                              original_primal,
                                                                                              config[
                                                                                                  'sparsification_sa'],
                                                                                              model_to_use='primal')

            log.info(f"{datetime.now()}: Processing dual...")

            eps_d, of_d, dv_d, ind_d, cviol_d, of_dec_d = sparsification_sensitivity_analysis(current_matrices_path,
                                                                                              created_dual,
                                                                                              config[
                                                                                                  'sparsification_sa'],
                                                                                              model_to_use='dual')

            # Assign results to the 'primal' and 'dual' keys using dictionary unpacking
            sparsification_results[model_name]['primal'] = {
                'epsilon': eps_p,
                'objective_function': of_p,
                'decision_variables': dv_p,
                'changed_indices': ind_p,
                'constraint_violation': cviol_p,
                'of_original_decision': of_dec_p
            }

            sparsification_results[model_name]['dual'] = {
                'epsilon': eps_d,
                'objective_function': of_d,
                'decision_variables': dv_d,
                'changed_indices': ind_d,
                'constraint_violation': cviol_d,
                'of_original_decision': of_dec_d
            }

        if config['sparsification_sa']['plots']:
            title_1 = 'Sparsification'
            primal_sense = original_primal.ModelSense

            visual_join_sensitivity(eps_p, of_p, dv_p, cviol_p, of_d, dv_d, cviol_d, title_1, primal_sense)

        # ================== Sensitivity analysis on constraint's reduction for different thresholds ===================
        if config['euclidian_reduction_sensitive_analysis']['val']:
            eps_p2, of_p2, dv_p2, ind_p2, cviol_p2, of_dec_p2 = (
                constraint_distance_reduction_sensitivity_analysis(current_matrices_path, original_primal,
                                                                   config[
                                                                       'euclidian_reduction_sensitive_analysis'],
                                                                   model_to_use='primal'))

            eps_d2, of_d2, dv_d2, ind_d2, cviol_d2, of_dec_d2 = (
                constraint_distance_reduction_sensitivity_analysis(current_matrices_path, created_dual,
                                                                   config[
                                                                       'euclidian_reduction_sensitive_analysis'],
                                                                   model_to_use='dual'))

            title_2 = 'Constraints Reduction'

            visual_join_sensitivity(eps_p2, of_p2, dv_p2, cviol_p2, of_d2, dv_d2, cviol_d2, title_2, primal_sense)

        # ======================================== Creating the presolved model ========================================
        if config['create_presolve']:
            presolved_model = original_primal.presolve()

        # ============================================== Clearing models  ==============================================
        # Delete model instances
        del original_primal_bp, original_primal, created_primal, created_dual

    # Saving the dictionary
    dict2json(sparsification_results, 'sparsification_results.json')
