import numpy as np
import scripts.create_sentences_graph as create_sentences_graph
import scripts.find_connected_components as find_connected_components

def find_threshold(matrices_list, sentences_list, min_threshold=0, max_threshold=1, step=0.01):
    """
    Find the optimal threshold for graph creation.

    Parameters:
    matrices_list (list of arrays): List of similarity matrices.
    sentences_list (list of lists): List of sentences for each text.
    min_threshold (float): Minimum threshold.
    max_threshold (float): Maximum threshold.
    step (float): Step to increase the threshold in each iteration.

    Returns:
    float: The value of the optimal threshold
    """
       
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    
    optimal_threshold = min_threshold
    max_total_sum = 0

    for threshold in thresholds:
        graph_list = create_sentences_graph.create_graph(matrices_list, threshold, sentences_list)
        _, total_sum = find_connected_components.find_components(graph_list)
        
        if total_sum > max_total_sum:
            max_total_sum = total_sum
            optimal_threshold = threshold

    return optimal_threshold
