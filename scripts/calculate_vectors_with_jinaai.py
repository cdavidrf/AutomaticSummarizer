from transformers import AutoModel

def calculate_vectors(sentences_list):
    """
    Calculate Jina AI vectors for each list of sentences in sentences_list using the jinaai/jina-embeddings-v2-base-en model.

    Parameters:
    sentences_list (list of list of str): List of lists where each sublist contains preprocessed sentences.

    Returns:
    list of array: A list of Jina AI vectors for the corresponding sentences.
    """
    
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

    jina_ai_vectors = []

    for sentences in sentences_list:
        sentence_vectors = model.encode(sentences)
        jina_ai_vectors.append(sentence_vectors)

    return jina_ai_vectors
