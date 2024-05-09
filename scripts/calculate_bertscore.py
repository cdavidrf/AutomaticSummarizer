from bert_score import BERTScorer

def calculate_scores(summaries_list, reference_summaries_list):
    """
    Calculate BERTScore metrics for a list of generated summaries compared to a list of reference summaries.

    Parameters:
    summary_list (list of str): List of generated summaries.
    reference_summary_list (list of str): List of reference summaries.

    Returns:
    list of tuple: A list of tuples where each tuple contains the BERTScore metrics for a pair of generated summary and reference summary.
    """
    
    bert_scores_list = []

    for (summary, reference_summary) in zip(summaries_list, reference_summaries_list):
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([summary], [reference_summary])
        bert_scores_list.append((P.mean(), R.mean(), F1.mean()))

    return bert_scores_list
