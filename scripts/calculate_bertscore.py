from bert_score import BERTScorer                                                                                                                           # type: ignore

def calculate_bert_scores(summary_list, reference_summary_list):
    """
    Calcula las métricas BERTScore para una lista de resúmenes generados en comparación con una lista de resúmenes de referencia.

    Parameters:
    summary_list (list of str): Lista de resúmenes generados.
    reference_summary_list (list of str): Lista de resúmenes de referencia.

    Returns:
    list of tuple: Una lista de tuplas donde cada tupla contiene las métricas BERTScore para un par de resumen generado y resumen de referencia.
    """
    
    print("\nComputing BERTScore for the summaries...\n")
    
    bert_scores_list = []  # Lista para almacenar los resultados de BERTScore

    # Bucle sobre cada par de resumen generado y resumen de referencia
    for i, (summary, reference_summary) in enumerate(zip(summary_list, reference_summary_list)):
        # Calcular BERTScore a partir de modelo
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([summary], [reference_summary])

        # Imprimir los resultados
        print(f"\nResultados para el resumen {i+1}:\n")
        print(f"BERTScore: Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
        print("\n--------------------------------------")

        bert_scores_list.append((P.mean(), R.mean(), F1.mean()))  # Agregar las métricas BERTScore a la lista de resultados

    return bert_scores_list



