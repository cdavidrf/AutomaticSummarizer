from sklearn.metrics.pairwise import cosine_similarity

def calculate_matrices(vectors):
    """
    Calculate the cosine similarity matrix for a set of vectors.

    Parameters:
    vectors (list of arrays): A list of vectors to compute similarity.

    Returns:
    list of arrays: A list of computed cosine similarity matrices.
    """
    
    cosine_matrices = []

    for vector in vectors:

        cosine_matrix = cosine_similarity(vector)
        cosine_matrices.append(cosine_matrix)

    return cosine_matrices
