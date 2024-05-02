from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

def fuse_sentences(
    sentences_lists,
    num_beams=1,
    num_beam_groups=1,
    num_return_sequences=1,
    diversity_penalty=0.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=2,
    temperature=1,
    max_length=256
):
    """
    Fusiona oraciones utilizando un modelo de generación de lenguaje.

    Parameters:
    sentences_lists (list of lists): Lista de listas de oraciones para fusionar.
    num_beams (int): Número de rayos a utilizar en la búsqueda de la secuencia.
    num_beam_groups (int): Número de grupos de rayos a usar en la búsqueda de la secuencia.
    num_return_sequences (int): Número de secuencias a generar para cada entrada.
    diversity_penalty (float): Penalización de diversidad aplicada a la distribución de palabras.
    repetition_penalty (float): Penalización de repetición aplicada a la distribución de palabras.
    no_repeat_ngram_size (int): Tamaño del n-grama para evitar repeticiones en la generación de secuencias.
    temperature (float): Controla la entropía en la generación de secuencias.
    max_length (int): Longitud máxima de la secuencia generada.

    Returns:
    list of lists: Lista de listas de oraciones fusionadas.
    """
    
    print("\nFusing sentences...\n")
    
    # Configuración del dispositivo
    device = "cpu"

    # Cargar el tokenizador y el modelo
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

    # Lista para almacenar las oraciones modificadas
    total_modified_sentences = []

    # Iterar sobre las listas de oraciones
    for i, sentences_list in enumerate(sentences_lists):
        #print(f"############# Text {i+1} ##############\n")
        modified_sentences = []

        # Iterar sobre las oraciones en la lista actual
        for j, sentence in enumerate(sentences_list):
            start_time = time.time()

            # Codificar la oración de entrada utilizando el tokenizador
            input_ids = tokenizer(
                f'paraphrase: {sentence}',
                return_tensors="pt", padding="longest",
                max_length=max_length,
                truncation=True,
            ).input_ids.to(device)

            # Generar la secuencia modificada utilizando el modelo
            outputs = model.generate(
                input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams, num_beam_groups=num_beam_groups,
                max_length=max_length, diversity_penalty=diversity_penalty
            )

            # Decodificar la secuencia generada
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            end_time = time.time()

            # Imprimir las oraciones originales y modificadas
            #print(f"Original Sentences {j+1}:\n{sentence}")
            #print(f"\nModified Sentences {j+1}:\n{res[0]}")
            #print(f"\nTime taken: {end_time-start_time} seconds")
            #print("--------------------------------------------")
            modified_sentences.append(res[0])

        total_modified_sentences.append(modified_sentences)

    return total_modified_sentences