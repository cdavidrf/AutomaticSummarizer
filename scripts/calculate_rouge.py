from rouge import Rouge

def calculate_rouge_scores(abstractive_summaries, reference_summaries):
    """
    Calcula las métricas ROUGE para los resúmenes abstractivos.

    Parameters:
    abstractive_summaries (list of str): Lista de resúmenes abstractivos.
    reference_summaries (list of str): Lista de resúmenes de referencia.

    Returns:
    list of dict: Lista de diccionarios con las métricas ROUGE para cada resumen.
    """
    
    print("\nComputing ROUGE scores for the summaries...\n")
    
    # Lista para almacenar los resultados ROUGE
    rouge_scores_list = []

    # Iterar sobre los resúmenes abstractivos
    for i, summary in enumerate(abstractive_summaries):
        # Obtener el resumen de referencia correspondiente
        reference_summary = reference_summaries[i].lower()

        # Crear un objeto Rouge
        rouge = Rouge()

        # Calcular las métricas ROUGE
        scores = rouge.get_scores(summary, reference_summary)

        # Agregar los resultados a la lista
        rouge_scores_list.append(scores)

        # Imprimir los resultados
        print(f"\nResultados para el resumen {i+1}:\n")
        for item in scores:
            for key, value in item.items():
                print(f"{key.capitalize()}: R = {value['r']:.4f}, P = {value['p']:.4f}, F = {value['f']:.4f}")

        print("\n--------------------------------------")

    return rouge_scores_list
