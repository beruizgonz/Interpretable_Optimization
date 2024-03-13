from datetime import datetime
import logging
from Interpretable_Optimization.models.utils_models.presolve_class import PresolveComillas
from collections import defaultdict
import gurobipy as gp
import sys
import os
from Interpretable_Optimization.models.utils_models.presolve_class import PresolveComillas
from Interpretable_Optimization.models.utils_models.utils_functions import nested_dict, get_model_matrices, \
    measuring_constraint_infeasibility, save_json, build_model_from_json
import numpy as np

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')


class SensitivityAnalysis:

    def __init__(self,
                 model=None,
                 save_path=None,
                 practical_infinity=None,
                 perform_reduction_small_coefficients=None):

        """
        docstring
        """
        self.model = model
        self.save_path = save_path
        self.practical_infinity = practical_infinity
        self.perform_reduction_small_coefficients = perform_reduction_small_coefficients

        self.sa_dictionary = defaultdict(nested_dict)

    def load_model_matrices(self):
        """
        Extract and load matrices and other components from the optimization model.
        """
        if self.model is None:
            raise ValueError("Model is not provided.")

        self.A, self.b, self.c, self.co, self.lb, self.ub, self.of_sense, self.cons_senses, self.variable_names = (
            get_model_matrices(self.model))

        # Generate original row indices from 0 to the number of constraints
        self.original_row_index = list(range(self.A.A.shape[0]))

        # Generate original columns indices from 0 to the number of variables
        self.original_column_index = list(range(self.A.A.shape[1]))

    def orchestrator_sensitivity_operations(self):
        """
        Perform sensitivity analysis based on the specified configurations.
        """

        # Ensure matrices are loaded
        self.load_model_matrices()

        sparsification_results = defaultdict(nested_dict)

        if self.perform_reduction_small_coefficients['val']:
            log.info(
                f"{str(datetime.now())}: Sensitivity Analysis: Small Coefficients")
            presolve_instance = PresolveComillas(model=self.model,
                                                 perform_reduction_small_coefficients={"val": True,
                                                                                       "threshold_small": None},
                                                 perform_bound_strengthening=False
                                                 )
            eps, of, dv, changed_indices, constraint_viol, of_dec = self.sens_anals_small_coeffs(presolve_instance)
            sparsification_results = {
                'epsilon': eps,
                'objective_function': of,
                'decision_variables': dv,
                'changed_indices': changed_indices,
                'constraint_violation': constraint_viol,
                'of_original_decision': of_dec
            }

        return sparsification_results

    def sens_anals_small_coeffs(self, presolve_instance):

        # Initialize lists to store results
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        eps = [0]  # Start with 0 threshold
        dv = [np.array([var.x for var in self.model.getVars()])]  # Start with decision variables of original model
        changed_indices = [None]  # List to store indices changed at each threshold
        abs_vio, obj_val = measuring_constraint_infeasibility(self.model, dv[0])
        of = [obj_val]  # Start with the objective value of the original model
        constraint_viol = [abs_vio]  # List to store infeasibility results
        of_dec = [self.model.objVal]
        iteration = 1  # Initialize iteration counter

        A_initial = self.A.A.copy()
        original_model = self.model.copy()

        # Calculate total iterations
        total_iterations = (
                int((self.perform_reduction_small_coefficients['max_threshold'] -
                     self.perform_reduction_small_coefficients['init_threshold']) /
                    self.perform_reduction_small_coefficients['step_threshold']) + 1)

        threshold = self.perform_reduction_small_coefficients['init_threshold']

        while threshold <= self.perform_reduction_small_coefficients['max_threshold']:
            presolve_instance.perform_reduction_small_coefficients['threshold_small'] = threshold
            (self.A, self.b, self.c, self.lb, self.ub, self.of_sense, self.cons_senses, self.co, self.variable_names,
             changes_dictionary, operation_table) = (presolve_instance.orchestrator_presolve_operations())

            A_changed = self.A.A.copy()

            # Record indices where A has been changed
            indices = [(i, j) for i in range(self.A.shape[0]) for j in range(self.A.shape[1]) if
                       A_initial[i, j] != 0 and A_changed[i, j] == 0]

            # Save the matrices
            save_json(self.A, self.b, self.c, self.lb, self.ub, self.of_sense, self.cons_senses, self.save_path,
                      self.co, self.variable_names)

            # Create a new model from the saved matrices
            iterative_model = build_model_from_json(self.save_path)

            # Solve the new model
            iterative_model.setParam('OutputFlag', 0)
            iterative_model.optimize()

            # Update lists with results
            # Check if the model found a solution
            if iterative_model.status == gp.GRB.OPTIMAL:
                # Model found a solution
                eps.append(threshold)
                of.append(iterative_model.objVal)
                changed_indices.append(indices)
                dv.append(np.array([var.x for var in iterative_model.getVars()]))
                # Access the decision variables
                variables = iterative_model.getVars()
                # Extract their values and store in a NumPy array
                decisions = np.array([var.x for var in variables])
                # calculate the constraint infeasibility
                abs_vio, obj_val = measuring_constraint_infeasibility(original_model, decisions)

                # Store infeasibility results
                of_dec.append(obj_val)
                constraint_viol.append(abs_vio)
            else:
                # Model did not find a solution
                eps.append(threshold)
                of.append(np.nan)  # Append NaN for objective function
                of_dec.append(np.nan)
                changed_indices.append(np.nan)
                constraint_viol.append(np.nan)
                dv.append(np.full(len(original_model.getVars()), np.nan))

            # Delete the model to free up resources
            del iterative_model

            # Progress bar
            progress = iteration / total_iterations
            bar_length = 50  # Length of the progress bar
            progress_bar = '>' * int(bar_length * progress) + ' ' * (bar_length - int(bar_length * progress))
            sys.stdout.write(f"\r[{progress_bar}] Iteration {iteration}/{total_iterations}")
            sys.stdout.flush()

            iteration += 1  # Increment iteration counter

            # Increment threshold
            threshold += self.perform_reduction_small_coefficients['step_threshold']
        return eps, of, dv, changed_indices, constraint_viol, of_dec

#         # =================== Sensitivity analysis on matrix sparsification for different thresholds ===================
#         if config['sparsification_sa']['val']:
#             log.info(
#                 f"{str(datetime.now())}: Sensitivity analysis on matrix sparsification for different thresholds:")
#
#         log.info(f"{datetime.now()}: Processing primal...")
#         eps_p, of_p, dv_p, ind_p, cviol_p, of_dec_p = sparsification_sensitivity_analysis(current_matrices_path,
#                                                                                           original_primal,
#                                                                                           config[
#                                                                                               'sparsification_sa'],
#                                                                                           model_to_use='primal')
#
#         log.info(f"{datetime.now()}: Processing dual...")
#
#         eps_d, of_d, dv_d, ind_d, cviol_d, of_dec_d = sparsification_sensitivity_analysis(current_matrices_path,
#                                                                                           created_dual,
#                                                                                           config[
#                                                                                               'sparsification_sa'],
#                                                                                           model_to_use='dual')
#
#         # Assign results to the 'primal' and 'dual' keys using dictionary unpacking
#         sparsification_results[model_name]['primal'] = {
#
#             'epsilon': eps_p,
#             'objective_function': of_p,
#             'decision_variables': dv_p,
#             'changed_indices': ind_p,
#             'constraint_violation': cviol_p,
#             'of_original_decision': of_dec_p
#
#         }
#
#
# sparsification_results[model_name]['dual'] = {
#     'epsilon': eps_d,
#     'objective_function': of_d,
#     'decision_variables': dv_d,
#     'changed_indices': ind_d,
#     'constraint_violation': cviol_d,
#     'of_original_decision': of_dec_d
# }

# if config['presolve_operations']['eliminate_zero_rows']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate zero rows")
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_zero_rows = eliminate_zero_rows(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_zero_columns']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate zero columns")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_zero_columns = eliminate_zero_columns(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_singleton_equalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate singleton equalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, solution_singleton_equalities = eliminate_singleton_equalities(current_model,
#                                                                                   current_matrices_path)
#
# if config['presolve_operations']['eliminate_doubleton_equalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate doubleton equalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model = eliminate_doubleton_equalities(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_kton_equalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate kton equalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, kton_dict = eliminate_kton_equalities(current_model, current_matrices_path, 3)
#
# if config['presolve_operations']['eliminate_singleton_inequalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate singleton inequalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint_single_in, feedback_variable_single_in = eliminate_singleton_inequalities(
#         current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_dual_singleton_inequalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate dual singleton inequalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint_single_in, feedback_variable_single_in = (
#         eliminate_dual_singleton_inequalities(current_model, current_matrices_path))
#
# if config['presolve_operations']['eliminate_redundant_columns']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate redundant columns")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint_red_col, feedback_variable_red_col = (
#         eliminate_redundant_columns(current_model, current_matrices_path))
#
# if config['presolve_operations']['eliminate_redundant_rows']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate redundant rows")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint = eliminate_redundant_rows(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_implied_bounds']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate implied bounds")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint = eliminate_implied_bounds(current_model, current_matrices_path)
#
# if config['presolve_operations']['small_coefficient_reduction']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - small coefficient reduction")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, changes = small_coefficient_reduction(current_model)
#
# # ============================================ rhs sensitivity analysis ========================================
# if config['rhs_sensitivity']:
#     log.info(
#         f"{str(datetime.now())}: Right-hand-side sensitivity analysis:")
#     rhs_dec, rhs_inc = rhs_sensitivity(original_primal_bp)
#
# # ======================================== cost vector sensitivity analysis ====================================
# if config['cost_sensitivity']:
#     log.info(
#         f"{str(datetime.now())}: Cost vector sensitivity analysis:")
#     cv_dec, cv_inc = cost_function_sensitivity(original_primal_bp)
#
# # ================================================ Test Sparsification =========================================
# if config['test_sparsification']['val']:
#     log.info(
#         f"{str(datetime.now())}: Results of the sparsification test:")
#     sparsification_test(original_primal, config, current_matrices_path)
#
# # ============================================ Test constraint reduction =======================================
# if config['test_constraint_red']['val']:
#     log.info(
#         f"{str(datetime.now())}: Results of the constraint reduction test:")
#     constraint_reduction_test(original_primal, config, current_matrices_path)
#
#     # =================== Sensitivity analysis on matrix sparsification for different thresholds ===================
#         if config['sparsification_sa']['val']:
#             log.info(
#                 f"{str(datetime.now())}: Sensitivity analysis on matrix sparsification for different thresholds:")
#
#             log.info(f"{datetime.now()}: Processing primal...")
#             eps_p, of_p, dv_p, ind_p, cviol_p, of_dec_p = sparsification_sensitivity_analysis(current_matrices_path,
#                                                                                               original_primal,
#                                                                                               config[
#                                                                                                   'sparsification_sa'],
#                                                                                               model_to_use='primal')
#
#             log.info(f"{datetime.now()}: Processing dual...")
#
#             eps_d, of_d, dv_d, ind_d, cviol_d, of_dec_d = sparsification_sensitivity_analysis(current_matrices_path,
#                                                                                               created_dual,
#                                                                                               config[
#                                                                                                   'sparsification_sa'],
#                                                                                               model_to_use='dual')
#
#             # Assign results to the 'primal' and 'dual' keys using dictionary unpacking
#             sparsification_results[model_name]['primal'] = {
#                 'epsilon': eps_p,
#                 'objective_function': of_p,
#                 'decision_variables': dv_p,
#                 'changed_indices': ind_p,
#                 'constraint_violation': cviol_p,
#                 'of_original_decision': of_dec_p
#             }
#
#             sparsification_results[model_name]['dual'] = {
#                 'epsilon': eps_d,
#                 'objective_function': of_d,
#                 'decision_variables': dv_d,
#                 'changed_indices': ind_d,
#                 'constraint_violation': cviol_d,
#                 'of_original_decision': of_dec_d
#             }
#
#         if config['sparsification_sa']['plots']:
#             title_1 = 'Sparsification'
#             primal_sense = original_primal.ModelSense
#
#             visual_join_sensitivity(eps_p, of_p, dv_p, cviol_p, of_d, dv_d, cviol_d, title_1, primal_sense)
#
#         # ================== Sensitivity analysis on constraint's reduction for different thresholds ===================
#         if config['euclidian_reduction_sensitive_analysis']['val']:
#             eps_p2, of_p2, dv_p2, ind_p2, cviol_p2, of_dec_p2 = (
#                 constraint_distance_reduction_sensitivity_analysis(current_matrices_path, original_primal,
#                                                                    config[
#                                                                        'euclidian_reduction_sensitive_analysis'],
#                                                                    model_to_use='primal'))
#
#             eps_d2, of_d2, dv_d2, ind_d2, cviol_d2, of_dec_d2 = (
#                 constraint_distance_reduction_sensitivity_analysis(current_matrices_path, created_dual,
#                                                                    config[
#                                                                        'euclidian_reduction_sensitive_analysis'],
#                                                                    model_to_use='dual'))
#
#             title_2 = 'Constraints Reduction'
#
#             visual_join_sensitivity(eps_p2, of_p2, dv_p2, cviol_p2, of_d2, dv_d2, cviol_d2, title_2, primal_sense)
#
#         # ======================================== Creating the presolved model ========================================
#         if config['create_presolve']:
#             presolved_model = original_primal.presolve()
#
#         # ============================================== Clearing models  ==============================================
#         # Delete model instances
#         del original_primal_bp, original_primal, created_primal, created_dual
#
#     # Saving the dictionary
#     dict2json(sparsification_results, 'sparsification_results.json')
#
# "rhs_sensitivity": False,
#               "cost_sensitivity": False,
#               "presolve_operations": {"eliminate_zero_rows": False,
#                                       "eliminate_zero_columns": False,
#                                       "eliminate_singleton_equalities": False,
#                                       "eliminate_doubleton_equalities": False,
#                                       "eliminate_kton_equalities": False,
#                                       "eliminate_singleton_inequalities": False,
#                                       "eliminate_dual_singleton_inequalities": False,
#                                       "eliminate_redundant_columns": False,
#                                       "eliminate_redundant_rows": False,
#                                       "eliminate_implied_bounds": True},
#               "test_sparsification": {"val": False,
#                                       "threshold": 0.13},
#               "test_constraint_red": {"val": False,
#                                       "threshold": 0.8},
#               "sparsification_sa": {"val": False,
#                                     "plots": False,
#                                     "prints": False,
#                                     "max_threshold": 0.3,
#                                     "init_threshold": 0.01,
#                                     "step_threshold": 0.001},
#               "euclidian_reduction_sensitive_analysis": {"val": False,
#                                                          "max_threshold": 1.4,
#                                                          "init_threshold": 0.1,
#                                                          "step_threshold": 0.01},
#               "create_presolve": False
