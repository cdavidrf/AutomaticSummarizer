import networkx as nx

def create_graph(matrices_list, similarity_threshold, sentences_list):
    """
    Create graphs from lists of similarity matrices.

    Parameters:
    matrices_list (list of arrays): List of similarity matrices.
    similarity_threshold (float): Similarity threshold to add an edge to the graph.
    sentences_list (list of lists): List of lists of sentences for each similarity matrix.

    Returns:
    list of networkx.Graph: List of created graphs.
    """
    
    graph_list = []

    for i, matrix in enumerate(matrices_list):
        G = nx.Graph()
        sentences = sentences_list[i]
        for j in range(len(sentences)):
            for k in range(j + 1, len(sentences)):
                similarity = matrix[j][k]
                if similarity > similarity_threshold:
                    G.add_edge(sentences[j], sentences[k], weight=similarity)

        graph_list.append(G)

    return graph_list
