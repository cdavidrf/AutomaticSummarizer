import cohere
import os, time
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def calculate_cohere_vectors(total_sentences):
    """
    Calcula los vectores Cohere para cada lista de oraciones en total_sentences utilizando la API de Cohere.

    Parameters:
    total_sentences (list of list of str): Lista de listas donde cada sublista contiene oraciones preprocesadas.
    cohere_api_key (str): Clave de API de Cohere.

    Returns:
    list of array: Una lista de vectores Cohere para las oraciones correspondientes.
    float: Tiempo transcurrido en segundos para calcular los vectores Cohere.
    """
    
    print("\nCalculating Cohere vectors...\n")

    start = time.time()  # Guardar el tiempo de inicio del proceso

    co = cohere.Client(COHERE_API_KEY)  # Inicializar el cliente de Cohere

    cohere_vectors = []  # Lista para almacenar los vectores Cohere

    # Iterar sobre cada lista de oraciones en total_sentences
    for sentences in total_sentences:
        # Realizar una solicitud a la API de Cohere para obtener los vectores de embeddings
        response = co.embed(texts=sentences, input_type="search_query", model="embed-english-v3.0")
        cohere_emb = response.embeddings
        cohere_vectors.append(cohere_emb)  # Agregar los vectores Cohere a la lista

    end = time.time()  # Guardar el tiempo de finalización del proceso
    elapsed_time = end - start  # Calcular el tiempo transcurrido
    # print(f"Tiempo transcurrido: {elapsed_time} segundos")  # Imprimir el tiempo transcurrido

    return cohere_vectors



