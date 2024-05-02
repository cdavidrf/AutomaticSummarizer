from transformers import AutoModel
import time

def calculate_jina_ai_vectors(total_sentences):
    """
    Calcula los vectores Jina AI para cada lista de oraciones en total_sentences utilizando el modelo jinaai/jina-embeddings-v2-base-en.

    Parameters:
    total_sentences (list of list of str): Lista de listas donde cada sublista contiene oraciones preprocesadas.

    Returns:
    list of array: Una lista de vectores Jina AI para las oraciones correspondientes.
    float: Tiempo transcurrido en segundos para calcular los vectores Jina AI.
    """
    
    print("\nCalculating Jina AI vectors...\n")
    
    start = time.time()  # Guardar el tiempo de inicio del proceso

    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)  # Inicializar el modelo de Jina AI

    jina_ai_vectors = []  # Lista para almacenar los vectores Jina AI

    # Iterar sobre cada lista de oraciones en total_sentences
    for sentences in total_sentences:
        # Calcular los vectores de oraciones utilizando el modelo de Jina AI
        sentence_vectors = model.encode(sentences)
        jina_ai_vectors.append(sentence_vectors)  # Agregar los vectores Jina AI a la lista

    end = time.time()  # Guardar el tiempo de finalización del proceso
    elapsed_time = end - start  # Calcular el tiempo transcurrido
    # print(f"Tiempo transcurrido: {elapsed_time} segundos")  # Imprimir el tiempo transcurrido

    return jina_ai_vectors, elapsed_time  





