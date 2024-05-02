import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_matrix(vectors):
    """
    Calcula la matriz de similitud de coseno para un conjunto de vectores.

    Parameters:
    vectors (list of arrays): Una lista de vectores para calcular la similitud.

    Returns:
    list of arrays: Una lista de matrices de similitud de coseno calculadas.
    """
    
    print("\nComputing cosine similarity matrices...\n")
    
    cosine_matrices = []  # Lista para almacenar las matrices de similitud de coseno

    for i, vector in enumerate(vectors):
        # Calcular la matriz de similitud de coseno para el vector actual
        cosine_matrix = cosine_similarity(vector)
        
        # Calcular el promedio de similitud del coseno para cada oración de manera eficiente
        average_similarity = np.mean(cosine_matrix, axis=1)

        # Calcular las estadísticas globales de similitud
        max_global_score = np.max(average_similarity)
        min_global_score = np.min(average_similarity)
        mean_global_score = np.mean(average_similarity)
        
        # Imprimir estadísticas globales
        # print(f"\nPuntuación máxima global para matriz {i+1}: {max_global_score}")
        # print(f"Puntuación mínima global para matriz {i+1}: {min_global_score}")
        # print(f"Puntuación promedio global para matriz {i+1}: {mean_global_score}")

        cosine_matrices.append(cosine_matrix)  # Agregar la matriz de similitud a la lista

    return cosine_matrices

