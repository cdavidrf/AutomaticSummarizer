import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def select_sentences(components_list, lambda_param=0.7):
    """
    Selects sentences using the Maximal Marginal Relevance (MMR) algorithm on a list of connected components.

    Parameters:
    components_list (list of sets): List of connected components.
    lambda_param (float): Lambda parameter for MMR calculation. Defaults to 0.7.

    Returns:
    tuple: A tuple containing a list of generated summaries, a list of concatenated sentences for each text,
           and a list of lists of selected sentences for each text.
    """
    
    extractive_summaries = []
    separated_sentences_lists = []
    concatenated_sentences_lists = []

    for components in components_list:
        separated_sentences_list = []
        concatenated_sentences_list = []

        for component in components:
            concatenated_sentences = ""

            if len(component) > 1:
                array_resultante = list(component)
                tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
                dt_matrix = tv.fit_transform(array_resultante)
                dt_matrix = dt_matrix.toarray()
                similarities = cosine_similarity(dt_matrix)
                selected_sentences = set()
                idx_max_similarity = np.argmax(np.sum(similarities, axis=0))
                selected_sentences.add(array_resultante[idx_max_similarity])
                
                for _ in range(1):

                    mmr_scores = []

                    for i, sentence in enumerate(array_resultante):

                        if sentence not in selected_sentences:
                            mmr_score = np.sum(similarities[i, list(map(lambda x: array_resultante.index(x), selected_sentences))]) - lambda_param * max(similarities[i, :])
                            mmr_scores.append((mmr_score, i))

                    if mmr_scores:        
                        selected_idx = max(mmr_scores, key=lambda x: x[0])[1]
                        selected_sentences.add(array_resultante[selected_idx])

                concatenated_sentences = '\n'.join(selected_sentences)
                concatenated_sentences_list.append(concatenated_sentences)
                separated_sentences_list.append(list(selected_sentences))

        separated_sentences_lists.append(separated_sentences_list)
        concatenated_sentences_lists.append(concatenated_sentences_list)
        extractive_summary = '\n'.join(concatenated_sentences_list)
        extractive_summaries.append(extractive_summary)

    return extractive_summaries, concatenated_sentences_lists, separated_sentences_lists
