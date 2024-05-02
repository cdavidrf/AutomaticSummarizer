import numpy as np

# Importar scripts
import scripts.create_graph as create_graph
import scripts.find_connected_components as find_connected_components

def find_optimal_threshold(matrices_list, total_sentences, min_threshold=0, max_threshold=1, step=0.01):
    """
    Encuentra el umbral óptimo para la creación de grafos.

    Parameters:
    matrices_list (list of arrays): Lista de matrices de similitud.
    min_threshold (float): Umbral mínimo.
    max_threshold (float): Umbral máximo.
    step (float): Paso para incrementar el umbral en cada iteración.
    total_sentences (list of lists): Lista de oraciones para cada texto.

    Returns:
    tuple: Una tupla que contiene el umbral óptimo y la suma total máxima de componentes conectados.
    """
    
    print("\nFinding optimal threshold...\n")

    # Generar una lista de umbrales desde min_threshold hasta max_threshold con un paso de 'step'
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    
    # Inicializar el umbral óptimo como el valor mínimo de umbral
    optimal_threshold = min_threshold
    
    # Inicializar la suma total máxima como cero
    max_suma_total = 0

    # Iterar sobre los umbrales y encontrar el umbral óptimo
    for threshold in thresholds:
        graph_list = create_graph.create_sentence_graph(matrices_list, threshold, total_sentences)
        _, suma_total = find_connected_components.connected_components(graph_list)
        #print(f"Umbral: {threshold}")
        #print("---------------------")
        
        # Actualizar el umbral óptimo si la suma total actual es mayor que la máxima encontrada hasta el momento
        if suma_total > max_suma_total:
            max_suma_total = suma_total
            optimal_threshold = threshold

    # print(f"\nOptimal Threshold: {optimal_threshold}")
    # print(f"Max Suma Total: {max_suma_total}")

    return optimal_threshold, max_suma_total
