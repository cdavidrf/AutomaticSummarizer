import cohere
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def calculate_vectors(sentences_list):
    """
    Calculate Cohere vectors for each list of sentences in sentences_list using the Cohere API.

    Parameters:
    sentences_list (list of list of str): List of lists where each sublist contains preprocessed sentences.

    Returns:
    list of array: A list of Cohere vectors for the corresponding sentences.
    """

    co = cohere.Client(COHERE_API_KEY)

    cohere_vectors = []

    for sentences in sentences_list:
        response = co.embed(texts=sentences, input_type="search_query", model="embed-english-v3.0")
        cohere_emb = response.embeddings
        cohere_vectors.append(cohere_emb)

    return cohere_vectors
