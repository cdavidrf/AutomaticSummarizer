import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def select_sentences(components_list, lambda_param=0.7):
    """
    Selecciona oraciones utilizando el algoritmo Maximal Marginal Relevance (MMR) en una lista de componentes conectados.

    Parameters:
    components_list (list of sets): Lista de componentes conectados.
    lambda_param (float): Parámetro lambda para el cálculo de MMR. Por defecto, 0.7.

    Returns:
    tuple: Una tupla que contiene una lista de resúmenes generados y una lista de listas de oraciones seleccionadas para cada texto.
    """
    
    print("\nSelecting sentences using MMR...\n")
    
    summaries = []  # Lista para almacenar los resúmenes generados
    all_texts_sentences = []  # Lista para almacenar todas las listas de oraciones seleccionadas

    for j, components in enumerate(components_list):
        #print(f"Text {j+1}")
        text_sentences = []  # Lista para almacenar las oraciones seleccionadas para cada texto
        for component in components:
            if len(component) > 1:
                array_resultante = list(component)

                # Vectorizar las oraciones usando TF-IDF
                tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
                dt_matrix = tv.fit_transform(array_resultante)
                dt_matrix = dt_matrix.toarray()

                # Calcular la similitud coseno entre cada par de oraciones
                similitudes = cosine_similarity(dt_matrix)

                # Inicializar el conjunto de oraciones seleccionadas
                selected_sentences = set()

                # Seleccionar la oración más relevante (mayor similitud)
                idx_max_similarity = np.argmax(np.sum(similitudes, axis=0))
                selected_sentences.add(array_resultante[idx_max_similarity])

                # Calcular MMR para seleccionar oraciones adicionales
                for _ in range(1):
                    mmr_scores = []
                    for i, oracion in enumerate(array_resultante):
                        if oracion not in selected_sentences:
                            # Calcular MMR score (relevancia - lambda * diversidad)
                            mmr_score = np.sum(similitudes[i, list(map(lambda x: array_resultante.index(x), selected_sentences))]) - lambda_param * max(similitudes[i, :])
                            mmr_scores.append((mmr_score, i))
                    if mmr_scores:
                        # Seleccionar la oración con el mayor MMR score
                        selected_idx = max(mmr_scores, key=lambda x: x[0])[1]
                        selected_sentences.add(array_resultante[selected_idx])
                        #print(f"Selected Sentences: \n{selected_sentences}")

                # Unir las oraciones seleccionadas en un solo string
                selected_sentence = '\n'.join(selected_sentences)
                text_sentences.append(selected_sentence)

        all_texts_sentences.append(text_sentences)
        complete_summary = '\n'.join(text_sentences)
        summaries.append(complete_summary)

    return summaries, all_texts_sentences
