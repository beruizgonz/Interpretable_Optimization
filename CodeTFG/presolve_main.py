import json
import os
from auxiliary_functions import *
from statistics import *
from indexes import *


def sensitivity_analysis(folder, modelo, datos, operation='Sparsification'):
    if modelo in datos:
        modelo_datos = datos[modelo]
        modelo_primal = modelo_datos.get('primal', {})
        modelo_pr_eps = modelo_primal.get('epsilon', [])

    optimaility = optimality_index(modelo, datos)
    print('Optimality')
    infeasibility = infeasibility_index(modelo, datos)
    print('Infeasibility')
    complexity = complexity_index_sparse(modelo, datos)
    plot_subplots(folder, modelo, modelo_pr_eps, optimaility, infeasibility, complexity, 'Optimality Index', 'Infeasibility Index', 'Complexity Index', operation)
    return


def global_sensitivity_analysis(models, data):
    # Define variables to store the data
    all_epsilons = {}
    all_objective_function_degradation = {}
    all_infeasibility = {}
    all_complexity = {}

    # Initialize lists to accumulate data
    objective_function_degradation_list = []
    infeasibility_list = []
    complexity_list = []

    # Determine the maximum length
    max_length = 0

    for model in models:
        if model in data:
            model_data = data[model]
            model_primal = model_data.get('primal', {})
            model_pr_eps = model_primal.get('epsilon', [])
            model_pr_of = model_primal.get('objective_function', [])
            model_pr_cv = model_primal.get('constraint_violation', [])
            model_pr_ci = model_primal.get('changed_indices', [])
            model_pr_ofod = model_primal.get('of_original_decision', [])
            model_du_dv = model_data.get('dual', {}).get('decision_variables', [])
            model_pr_dv = model_primal.get('decision_variables', [])

            # Calculation of original objective function degradation
            divisor = model_pr_of[0] if model_pr_of else None
            model_pr_ofod_pu = [element / divisor for element in model_pr_of] if divisor else []

            # Calculation of INFEASIBILITY INDEX
            reference_value = 1e-6
            model_pr_cv_no_nan = remove_nan_sublists(model_pr_cv)
            model_pr_cv_filtered = set_values_below_threshold_to_zero(model_pr_cv_no_nan, reference_value)

            if model_du_dv and model_pr_cv_filtered:
                product_cv_dv = multiply_matrices(model_pr_cv_filtered, model_du_dv)
                sum_product = sum_sublists(product_cv_dv)
                infeasibility_index = [x / abs(model_pr_ofod[0]) for x in sum_product] if model_pr_ofod else []
            else:
                infeasibility_index = []

            # Calculation of PROBLEM COMPLEXITY
            accumulated_model_ci = calculate_lengths(model_pr_ci)
            convert_late_zeros_to_nan(accumulated_model_ci)
            total_elements_A = len(model_pr_dv) * len(model_du_dv)
            elements_A_made_0_pu = [x / total_elements_A for x in accumulated_model_ci] if total_elements_A else []
            problem_complexity = [1 - x for x in elements_A_made_0_pu]

            # Update the maximum length
            max_length = max(max_length, len(model_pr_eps), len(model_pr_ofod_pu), len(infeasibility_index), len(problem_complexity))

            # Accumulate data for global analysis
            all_epsilons[model] = model_pr_eps
            all_objective_function_degradation[model] = model_pr_ofod_pu
            all_infeasibility[model] = infeasibility_index
            all_complexity[model] = problem_complexity

            # Ensure all lists have the same length
            objective_function_degradation_list.append(fill_with_nan(model_pr_ofod_pu, max_length))
            infeasibility_list.append(fill_with_nan(infeasibility_index, max_length))
            complexity_list.append(fill_with_nan(problem_complexity, max_length))
        else:
            print(f"The model '{model}' was not found in the provided data.")

    def calculate_statistics_by_index(data_by_index):
        num_epsilons = len(data_by_index[0]) if data_by_index else 0
        mean = [calculate_mean([data[i] for data in data_by_index]) for i in range(num_epsilons)]
        median = [calculate_median([data[i] for data in data_by_index]) for i in range(num_epsilons)]
        quartiles = [calculate_quartiles([data[i] for data in data_by_index]) for i in range(num_epsilons)]
        return mean, median, quartiles

    mean_of, median_of, quartiles_of = calculate_statistics_by_index(objective_function_degradation_list)
    mean_infeasibility, median_infeasibility, quartiles_infeasibility = calculate_statistics_by_index(infeasibility_list)
    mediana_complexity, mediana_complexity, cuartiles_complexity = calculate_statistics_by_index(complexity_list)
    # Function to plot the data
    def plot_data(title, ylabel, data_dict, statistics_dict=None):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.get_cmap('tab10', len(data_dict))

        for i, model in enumerate(models):
            eps = all_epsilons.get(model, [])
            data = data_dict.get(model, [])
            eps_adjusted, data_adjusted = adjust_lengths(eps, data)
            plt.plot(eps_adjusted, data_adjusted, label=f'{model}')

        if statistics_dict:
            num_epsilons = len(statistics_dict.get('mean', []))
            #common_eps = np.linspace(min(eps_adjusted), max(eps_adjusted), num=num_epsilons)
            mean = statistics_dict.get('mean', [])
            median = statistics_dict.get('median', [])
            quartiles = statistics_dict.get('quartiles', [])
            quartile_25 = [q[0] for q in quartiles]
            quartile_75 = [q[1] for q in quartiles]

        # plt.plot(eps_adjusted, mean[:len(eps_adjusted)], '--', label='Mean', color='black')
        # plt.plot(eps_adjusted, median[:len(eps_adjusted)], ':', label='Median', color='blue')
        # plt.plot(eps_adjusted, quartile_25[:len(eps_adjusted)], '-.', label='Quartile 25', color='green')
        # plt.plot(eps_adjusted, quartile_75[:len(eps_adjusted)], '-.', label='Quartile 75', color='red')

        figure_path = os.path.join(os.getcwd(), 'figures_sparsification')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.title(title, fontsize=20)
        plt.xlabel("Epsilon",  fontsize=14)
        plt.ylabel(ylabel)
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(figure_path, f"{title}.png"))
        plt.show()
    

# Assuming functions like remove_nan_sublists, set_values_below_threshold_to_zero, multiply_matrices, etc. are already defined.


    plot_data(f"Objective function degradation_{models}", f"Objective function_{models}", all_objective_function_degradation, {'mean': mean_of, 'median': median_of, 'quartiles': quartiles_of})
    plot_data(f"Infeasibility evolution_{models}", f"Infeasibility_{models}", all_infeasibility, {'mean': mean_infeasibility, 'median': median_infeasibility, 'quartiles': quartiles_infeasibility})
    plot_data(f"Complexity evolution_{models}", f"Complexity_{models}", all_complexity, {'mean': mediana_complexity, 'median': mediana_complexity, 'quartiles': cuartiles_complexity})
    # Save the figures


    return


if __name__ == '__main__': 
    # # Load the data from the JSON file
    # parent_dir = os.path.dirname(os.getcwd())

    # data_path = os.path.join(parent_dir, 'models/epsilon_sparsification.json')
    # print(os.path.exists(data_path))
    # with open(data_path, 'r') as file:
    #     data = json.load(file)

    # modelos_tipo1 = ['AIRSP', 'PRODMIX', 'SPARTA' ]
    # #graphed:
    # modelos_tipo2 = ['AIRCRAFT','SRKANDW','UIMP']
    # #grahped:
    # modelos_tipo3 = ['GUSSEX1','GUSSGRID','SENSTRAN','TRNSPORT']
    # modelos_tipo4 = ['PORT']

    # # Perform the global sensitivity analysis
    # # global_sensitivity_analysis(modelos_tipo4, data)
    # # global_sensitivity_analysis(modelos_tipo3, data)
    # # global_sensitivity_analysis(modelos_tipo2, data)
    # # global_sensitivity_analysis(modelos_tipo1, data)
    # sensitivity_analysis('AIRSP', data)
    project_root = os.path.dirname(os.getcwd())
    figures_folder = os.path.join(project_root, 'figures_new/global/sparse/gams_library')
    figures_rows_folder = os.path.join(figures_folder, 'epsilon_rows_norm')
    figures_cols_folder = os.path.join(figures_folder,'epsilon_cols_norm_flexibility')
    figures_sparsification = os.path.join(figures_folder, 'sparsification_flexibility')
    figures_dependency_rows = os.path.join(figures_folder, 'epsilon_dependency_rows')
    figures_dependency_cols = os.path.join(figures_folder, 'epsilon_dependency_cols')

    results_foder = os.path.join(project_root, 'results_new/global/sparse/gams_library')
    results_sparsification_folder = os.path.join(results_foder, 'prueba')
    results_rows_folder = os.path.join(results_foder, 'epsilon_rows_norm')
    results_cols_folder = os.path.join(results_foder, 'epsilon_cols_norm_flexibility')
    results_dependency_rows_folder = os.path.join(results_foder, 'epsilon_dependency_rows')
    results_dependency_cols_folder = os.path.join(results_foder, 'epsilon_dependency_cols')

    optimun_bounds_folder = os.path.join(results_foder, 'sparsification_optimum_bounds')
    figures_optimum_bounds = os.path.join(project_root, 'figures/sparsification_optimum_bounds')

    # if not os.path.exists(figures_optimum_bounds):
    #     os.makedirs(figures_optimum_bounds)

    # model = 'DINAM'
    # model = 'openTEPES_EAPP_2030_sc01_st1'
    # json_path = os.path.join(results_sparsification_folder, f'epsilon_sparsification_{model}_flexibility1.json')
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    # sensitivity_analysis(figures_sparsification, model, data)
    
    # for root, dirs, files in os.walk(results_sparsification_folder):
    #     operation = 'Sparsification'
    #     for file in files:
    #         if file.endswith('.json'):
    #             print(f"Processing file {file}")
    #             if 'MARCO' in file:
    #                 continue
    #             else:
    #                 with open(os.path.join(root, file), 'r') as f:
    #                     data = json.load(f)
    #                     for model in data.keys():  
    #                         sensitivity_analysis(figures_sparsification, model, data,operation)
    # for root, dirs, files in os.walk(results_dependency_rows_folder):
    #     operation = 'Dependency Rows'
    #     for file in files:
    #         if file.endswith('.json'):
    #             print(f"Processing file {file}")
    #             if 'MARCO' in file:
    #                 continue
    #             else:
    #                 with open(os.path.join(root, file), 'r') as f:
    #                     data = json.load(f)
    #                     for model in data.keys():  
    #                         sensitivity_analysis(figures_dependency_rows, model, data, operation)
    # for root, dirs, files in os.walk(results_dependency_cols_folder):
    #     operation = 'Dependency Cols'
    #     for file in files:
    #         if file.endswith('.json'):
    #             print(f"Processing file {file}")
    #             if 'MARCO' in file:
    #                 continue
    #             else:
    #                 with open(os.path.join(root, file), 'r') as f:
    #                     data = json.load(f)
    #                     for model in data.keys():  
    #                         sensitivity_analysis(figures_dependency_cols, model, data, operation)
    for root, dirs, files in os.walk(results_cols_folder):
        operation = 'Epsilon Cols'
        for file in files:
            if file.endswith('.json'):
                print(f"Processing file {file}")
                if 'MARCO' in file:
                    continue
                else:
                    with open(os.path.join(root, file), 'r') as f:
                        data = json.load(f)
                        for model in data.keys():  
                            sensitivity_analysis(figures_cols_folder, model, data, operation)
    # for root, dirs, files in os.walk(results_rows_folder):
    #     operation = 'Epsilon rows'
    #     for file in files:
    #         if file.endswith('.json'):
    #             print(f"Processing file {file}")
    #             if 'MARCO' in file:
    #                 continue
    #             else:
    #                 with open(os.path.join(root, file), 'r') as f:
    #                     data = json.load(f)
    #                     for model in data.keys():  
    #                         sensitivity_analysis(figures_rows_folder, model, data, operation)

    # data = json.load(open(os.path.join(results_rows_folder, 'epsilon_rows_CLEARLAK.json')))
    # #print(data)
    # sensitivity_analysis(figures_rows_folder, 'CLEARLAK', data)