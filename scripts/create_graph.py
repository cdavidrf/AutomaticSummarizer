import networkx as nx

def create_sentence_graph(matrices_list, umbral_similitud, total_sentences):
    """
    Crea grafos a partir de listas de matrices de similitud.

    Parameters:
    matrices_list (list of arrays): Lista de matrices de similitud.
    umbral_similitud (float): Umbral de similitud para agregar una arista al grafo.
    total_sentences (list of lists): Lista de listas de oraciones para cada matriz de similitud.

    Returns:
    list of networkx.Graph: Lista de grafos creados.
    """
    
    print("\nCreating sentence graphs...\n")
    
    graph_list = []  # Lista para almacenar los grafos generados

    for i, matrix in enumerate(matrices_list):
        G = nx.Graph()  # Crear un nuevo grafo no dirigido
        sentences = total_sentences[i]  # Obtener la lista de oraciones actual
        for j in range(len(sentences)):
            for k in range(j + 1, len(sentences)):
                similarity = matrix[j][k]  # Obtener la similitud entre las oraciones j y k
                if similarity > umbral_similitud:  # Comprobar si la similitud supera el umbral
                    # Añadir una arista ponderada al grafo entre las oraciones j y k
                    G.add_edge(sentences[j], sentences[k], weight=similarity)

        graph_list.append(G)  # Añadir el grafo a la lista de grafos generados

    return graph_list

