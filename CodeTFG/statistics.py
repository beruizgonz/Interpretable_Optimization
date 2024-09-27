import numpy as np
import matplotlib.pyplot as plt

def calculate_mean(lst):
    return np.nanmean(lst) if lst else None

def calculate_median(lst):
    return np.nanmedian(lst) if lst else None

def calculate_quartiles(lst):
    return np.nanpercentile(lst, [25, 75]) if lst else (None, None)

def adjust_lengths(eps, data):
    min_length = min(len(eps), len(data))
    return eps[:min_length], data[:min_length]

def fill_with_nan(data, max_length):
    return data + [np.nan] * (max_length - len(data))



#     # Función para calcular estadísticas por índice
# def calcular_estadisticas_por_indice(datos_por_indice):
#     num_epsilons = len(datos_por_indice[0]) if datos_por_indice else 0
#     media = [calcular_media([datos[i] for datos in datos_por_indice]) for i in range(num_epsilons)]
#     mediana = [calcular_mediana([datos[i] for datos in datos_por_indice]) for i in range(num_epsilons)]
#     cuartiles = [calcular_cuartiles([datos[i] for datos in datos_por_indice]) for i in range(num_epsilons)]
#     return media, mediana, cuartiles

# # Calcular estadísticas para cada métrica
# # media_of, mediana_of, cuartiles_of = calcular_estadisticas_por_indice(objective_function_degradation_list)
# # media_infeasibility, mediana_infeasibility, cuartiles_infeasibility = calcular_estadisticas_por_indice(infeasibility_list)
# # media_complexity, mediana_complexity, cuartiles_complexity = calcular_estadisticas_por_indice(complexity_list)

# # Función para graficar los datos
# def graficar_datos(titulo, ylabel, datos_dict, estadisticas_dict=None):
#     plt.figure(figsize=(10, 6))
#     colores = plt.cm.get_cmap('tab10', len(datos_dict))
    
#     for i, modelo in enumerate(modelos):
#         eps = all_epsilons.get(modelo, [])
#         datos = datos_dict.get(modelo, [])
#         eps_ajustado, datos_ajustado = ajustar_longitudes(eps, datos)
#         plt.plot(eps_ajustado, datos_ajustado, label=f'{modelo}')
    
#     if estadisticas_dict:
#         num_epsilons = len(estadisticas_dict.get('media', []))
#         eps_comunes = np.linspace(min(eps_ajustado), max(eps_ajustado), num=num_epsilons)
#         media = estadisticas_dict.get('media', [])
#         mediana = estadisticas_dict.get('mediana', [])
#         cuartiles = estadisticas_dict.get('cuartiles', [])
#         cuartiles_25 = [q[0] for q in cuartiles]
#         cuartiles_75 = [q[1] for q in cuartiles]

#         plt.plot(eps_comunes, media[:len(eps_comunes)], '--', label='Media', color='black')
#         plt.plot(eps_comunes, mediana[:len(eps_comunes)], ':', label='Mediana', color='blue')
#         plt.plot(eps_comunes, cuartiles_25[:len(eps_comunes)], '-.', label='Quartile 25', color='green')
#         plt.plot(eps_comunes, cuartiles_75[:len(eps_comunes)], '-.', label='Quartile 75', color='red')

#     plt.title(titulo, fontname='Times New Roman', fontsize=20)
#     plt.xlabel("Epsilon", fontname='Times New Roman', fontsize=14)
#     plt.ylabel(ylabel)
#     plt.legend(fontsize=12)
#     plt.show()

# # # Graficar datos acumulados
# # graficar_datos("Objective Function Degradation for type 4 models", "", all_objective_function_degradation, {'media': media_of, 'mediana': mediana_of, 'cuartiles': cuartiles_of})
# # graficar_datos("Infeasibility Evolution for type 4 models", "", all_infeasibility, {'media': media_infeasibility, 'mediana': mediana_infeasibility, 'cuartiles': cuartiles_infeasibility})
# # graficar_datos("Complexity Evolution for type 4 models", "", all_complexity, {'media': media_complexity, 'mediana': mediana_complexity, 'cuartiles': cuartiles_complexity})

# # return
