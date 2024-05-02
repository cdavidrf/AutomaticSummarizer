from sklearn.feature_extraction.text import TfidfVectorizer
import time

def calculate_tfidf_vectors(total_sentences):
    """
    Calcula los vectores TF-IDF para cada lista de oraciones en total_sentences.

    Parameters:
    total_sentences (list of list of str): Lista de listas donde cada sublista contiene oraciones preprocesadas.

    Returns:
    list of array: Una lista de matrices densas de NumPy donde cada matriz contiene los vectores TF-IDF de las oraciones correspondientes.
    float: Tiempo transcurrido en segundos para calcular los vectores TF-IDF.
    """
    
    print("\nCalculating TF-IDF vectors...\n")
    
    start = time.time()  # Guardar el tiempo de inicio del proceso

    tfidf_vectors = []  # Lista para almacenar los vectores TF-IDF

    # Iterar sobre cada lista de oraciones en total_sentences
    for sentences in total_sentences:
        # Inicializar y ajustar el TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()
        # Transformar las oraciones en vectores TF-IDF utilizando el TfidfVectorizer
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        # Convertir la matriz de TF-IDF a una matriz densa de NumPy
        tfidf_matrix = tfidf_matrix.toarray()
        # Agregar los vectores TF-IDF a la lista
        tfidf_vectors.append(tfidf_matrix)  

    end = time.time()  # Guardar el tiempo de finalización del proceso
    elapsed_time = end - start  # Calcular el tiempo transcurrido
    # print(f"Tiempo transcurrido: {elapsed_time} segundos")  # Imprimir el tiempo transcurrido

    return tfidf_vectors


