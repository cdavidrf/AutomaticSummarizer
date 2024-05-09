from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_vectors(sentences_list):
    """
    Calculate TF-IDF vectors for each list of sentences in sentences_list.

    Parameters:
    sentences_list (list of list of str): List of lists where each sublist contains preprocessed sentences.

    Returns:
    list of array: A list of NumPy dense matrices where each matrix contains TF-IDF vectors of the corresponding sentences.
    """
    
    tfidf_vectors = []

    for sentences in sentences_list:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        tfidf_matrix = tfidf_matrix.toarray()
        tfidf_vectors.append(tfidf_matrix)  

    return tfidf_vectors
