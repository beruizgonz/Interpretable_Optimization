import json
from .auxiliary_functions import *
from .statistics import *

def analisis_de_sensibilidad(modelo, datos):
    if modelo in datos:
        modelo_datos = datos[modelo]
        modelo_primal = modelo_datos.get('primal', {})
        modelo_dual = modelo_datos.get('dual', {})
        modelo_pr_eps = modelo_primal.get('epsilon', [])
        modelo_pr_of = modelo_primal.get('objective_function', [])
        modelo_pr_dv = modelo_primal.get('decision_variables', [])
        modelo_pr_ci = modelo_primal.get('changed_indices', [])
        modelo_pr_cv = modelo_primal.get('constraint_violation', [])
        modelo_pr_ofod = modelo_primal.get('of_original_decision', [])
        modelo_pr_time = modelo_primal.get('execution_time', [])
        modelo_du_dv = modelo_dual.get('decision_variables', [])
        
        # Cálculo de degradación de la función objetivo original
        divisor = modelo_pr_of[0] if modelo_pr_of else None
        modelo_pr_of_pu = [elemento / divisor for elemento in modelo_pr_of] if divisor else None
        
        # Cálculo de degradación de la función objetivo original
        divisor1 = modelo_pr_ofod[0] if modelo_pr_ofod else None
        modelo_pr_ofod_pu = [elemento / divisor1 for elemento in modelo_pr_ofod] if divisor1 else None
        
        ## Cálculo de INFEASIBILITY INDEX
        cifra_referencia = 1e-6
        modelo_pr_cv_sin_nan = remove_nan_sublists(modelo_pr_cv)
        modelo_pr_cv_filtrado = set_values_below_threshold_to_zero(modelo_pr_cv_sin_nan, cifra_referencia)
        producto_cv_vd = multiply_matrices(modelo_pr_cv_filtrado, modelo_du_dv)
        suma_producto = sum_sublists(producto_cv_vd)
        infeasiblity_index = [x / abs(modelo_pr_ofod[0]) for x in suma_producto]
        
        ## Cálculo de PROBLEM COMPLEXITY
        ## Cálculo de la media de la constraint violations
        modelo_pr_cv_medias = calculate_means(modelo_pr_cv)
        
        ## Cálculo de los índices cambiados a 0 de la matriz A
        acumulado_modelo_ci = calculate_lengths(modelo_pr_ci)
        # convert_late_zeros_to_nan(acumulado_modelo_ci) # ¿Qué es esta función?
        elementos_A_totales = len(modelo_pr_dv) * len(modelo_du_dv)
        
        ## Ahora calculamos el número de no 0s en la matriz A, para cada valor de epsilon.
        ## Esto lo podemos calcular mediante el número de indices que cambian en cada nivel de epsilon
        elementos_A_que_se_hacen_0_pu = [x / elementos_A_totales for x in acumulado_modelo_ci]
        complexity_problem = [1 - x for x in elementos_A_que_se_hacen_0_pu]
        
        suma_objfunc_unfeasiblity = [a + b for a, b in zip(modelo_pr_ofod_pu, infeasiblity_index)]
        
        ## PROBLEM COMPLEXITY, con el tiempo de ejecución (comentado en el código original)
        # modelo_pr_time1 = modelo_pr_time[1:]
        # #Convertir los tiempos a números flotantes
        # modelo_pr_time1 = [float(tiempo.split(':')[2]) for tiempo in modelo_pr_time1]
        # divisor = modelo_pr_time1[0] if modelo_pr_time1 else None
        # modelo_pr_time_pu = [elemento / divisor for elemento in modelo_pr_time1] if divisor else None
        # complexity_problem = modelo_pr_time_pu
        
        ## GRÁFICAS
        titulo1 = (modelo + " Objective function degradation")
        titulo2 = (modelo + " Infeasibility evolution")
        titulo3 = (modelo + " Complexity evolution")
        objective_function = "Objective function"
        complexity = "Complexity"
        infeasibility = "Infeasibility"
        
        plot1(modelo_pr_eps, modelo_pr_of_pu, titulo1, objective_function)
        plot1(modelo_pr_eps, infeasiblity_index, titulo2, infeasibility)
        plot1(modelo_pr_eps, complexity_problem, titulo3, complexity)
        
        print(elementos_A_que_se_hacen_0_pu)
        return
    else:
        print(f"El modelo '{modelo}' no se encontró en los datos proporcionados.")
        return None

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
            common_eps = np.linspace(min(eps_adjusted), max(eps_adjusted), num=num_epsilons)
            mean = statistics_dict.get('mean', [])
            median = statistics_dict.get('median', [])
            quartiles = statistics_dict.get('quartiles', [])
            quartile_25 = [q[0] for q in quartiles]
            quartile_75 = [q[1] for q in quartiles]

        plt.plot(common_eps, mean[:len(common_eps)], '--', label='Mean', color='black')
        plt.plot(common_eps, median[:len(common_eps)], ':', label='Median', color='blue')
        plt.plot(common_eps, quartile_25[:len(common_eps)], '-.', label='Quartile 25', color='green')
        plt.plot(common_eps, quartile_75[:len(common_eps)], '-.', label='Quartile 75', color='red')

        plt.title(title, fontsize=20)
        plt.xlabel("Epsilon",  fontsize=14)
        plt.ylabel(ylabel)
        plt.legend(fontsize=12)
        plt.show()
    
    mean_of, median_of, quartiles_of = calculate_statistics_by_index(objective_function_degradation_list)
    mean_infeasibility, median_infeasibility, quartiles_infeasibility = calculate_statistics_by_index(infeasibility_list)
    mediana_complexity, mediana_complexity, cuartiles_complexity = calculate_statistics_by_index(complexity_list)
# Assuming functions like remove_nan_sublists, set_values_below_threshold_to_zero, multiply_matrices, etc. are already defined.


    plot_data("Objective function degradation", "Objective function", all_objective_function_degradation, {'mean': mean_of, 'median': median_of, 'quartiles': quartiles_of})
    plot_data("Infeasibility evolution", "Infeasibility", all_infeasibility, {'mean': mean_infeasibility, 'median': median_infeasibility, 'quartiles': quartiles_infeasibility})
    plot_data("Complexity evolution", "Complexity", all_complexity, {'mean': mediana_complexity, 'median': mediana_complexity, 'quartiles': cuartiles_complexity})