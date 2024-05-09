from rouge import Rouge

def calculate_scores(abstractive_summaries, reference_summaries):
    """
    Calculate ROUGE metrics for the abstractive summaries.

    Parameters:
    abstractive_summaries (list of str): List of abstractive summaries.
    reference_summaries (list of str): List of reference summaries.

    Returns:
    list of dict: List of dictionaries with ROUGE metrics for each summary.
    """
    
    rouge_scores_list = []

    for i, summary in enumerate(abstractive_summaries):
        reference_summary = reference_summaries[i].lower()
        rouge = Rouge()
        scores = rouge.get_scores(summary, reference_summary)
        rouge_scores_list.append(scores)

    return rouge_scores_list
