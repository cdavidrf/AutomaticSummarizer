# Authors: Cristian David Rocha Fernández, Juan Carlos Montes Estrada, Andrés Felipe Hernández Giraldo
import pandas as pd
import time
import scripts.preprocess_transcripts as preprocess_transcripts
import scripts.calculate_vectors_with_tfidf as calculate_vectors_with_tfidf
import scripts.calculate_vectors_with_cohere as calculate_vectors_with_cohere
import scripts.calculate_vectors_with_jinaai as calculate_vectors_with_jinaai
import scripts.calculate_cosine_matrices as calculate_cosine_matrices
import scripts.find_optimal_threshold as find_optimal_threshold
import scripts.create_sentences_graph as create_sentences_graph
import scripts.find_connected_components as find_connected_components
import scripts.select_sentences_with_mmr as select_sentences_with_mmr
import scripts.fuse_sentences_with_model as fuse_sentences_with_model
import scripts.fuse_sentence_with_alignment as fuse_sentence_with_alignment
import scripts.calculate_rouge as calculate_rouge
import scripts.calculate_bertscore as calculate_bertscore

start = time.time()

#data = pd.read_csv("data/1sample.csv",sep=",")
data = pd.read_csv("data/4samples.csv",sep=",")

# Extract transcripts
transcripts = data['text']

# Preprocess transcripts
preprocessed_transcripts = preprocess_transcripts.preprocess(transcripts)

# Create vectors
tfidf_vectors = calculate_vectors_with_tfidf.calculate_vectors(preprocessed_transcripts)
# cohere_vectors = calculate_vectors_with_cohere.calculate_vectors(preprocessed_transcripts)
# jinaai_vectors = calculate_vectors_with_jinaai.calculate_vectors(preprocessed_transcripts)

# Calculate cosine similarities matrices
cosine_matrices = calculate_cosine_matrices.calculate_matrices(tfidf_vectors)

# Calculate optimal threshold
optimal_threshold = find_optimal_threshold.find_threshold(cosine_matrices, preprocessed_transcripts)

# Create graphs
graph_list = create_sentences_graph.create_graph(cosine_matrices, optimal_threshold, preprocessed_transcripts)

# Find connected components
components_list,_ = find_connected_components.find_components(graph_list)

# Seleccionar oraciones representativas utilizando MMR
extractive_summaries, concatenated_sentences_lists, separated_sentences_lists = select_sentences_with_mmr.select_sentences(components_list)

# Fuse sentences
fused_sentences = fuse_sentences_with_model.fuse_sentences(concatenated_sentences_lists)
fused_sentences2 = fuse_sentence_with_alignment.fuse_sentences(separated_sentences_lists)

# Print extractive summaries
for i, summary in enumerate(extractive_summaries):
  print(f"\nExtractive Summary {i+1}:\n\n{summary}")
  print("\n-------------------------------------------------------------")

# Concatenate sentences to generate summaries
summaries_method1=[]
for i, fused_sentences in enumerate(fused_sentences):
  summary = '\n'.join(fused_sentences)
  summaries_method1.append(summary)
  print(f"\nSummary {i+1} (Method 1):\n\n{summary}")
  print("\n------------------------------------------------------------------------------")

# Concatenar todas las oraciones de cada lista interna en una sola cadena
summaries_method2 = []

# Procesar cada lista interna de oraciones separadas
for i, fused_sentences in enumerate(fused_sentences2):
    # Concatenar todas las oraciones de la lista interna
    summary = '\n'.join(fused_sentence for fused_sentence in fused_sentences)
    print(f"\nSummary {i+1} (Method 2):\n\n{summary}")
    print("\n-------------------------------------------------------------------------------")
    summaries_method2.append(summary)  # Agregar el resumen final a la lista

# Calculate ROUGE Scores
reference_summaries=data['reference_summary']
rouge_scores1 = calculate_rouge.calculate_scores(extractive_summaries, reference_summaries)
rouge_scores2 = calculate_rouge.calculate_scores(summaries_method1, reference_summaries)
rouge_scores3 = calculate_rouge.calculate_scores(summaries_method2, reference_summaries)

# Calculate BERT Scores
bert_scores1 = calculate_bertscore.calculate_scores(extractive_summaries, reference_summaries)
bert_scores2 = calculate_bertscore.calculate_scores(summaries_method1, reference_summaries)
bert_scores3 = calculate_bertscore.calculate_scores(summaries_method2, reference_summaries)

# Print ROUGE and BERT Scores
print(f"\n##### ROUGE Scores #####\n")
for i, sublist in enumerate(rouge_scores1):
    print(f"\nExtractive Summary {i+1}:\n")
    for dictionary in sublist:
        for metric, values in dictionary.items():
            print(f"\tMetric: {metric}")
            for sub_metric, score in values.items():
                print(f"\t{sub_metric}: {score}\n")
print("-----------------------------------------")
for i, sublist in enumerate(rouge_scores2):
    print(f"\nSummary {i+1} (Method 1):\n")
    for dictionary in sublist:
        for metric, values in dictionary.items():
            print(f"\tMetric: {metric}")
            for sub_metric, score in values.items():
                print(f"\t{sub_metric}: {score}\n")
print("-----------------------------------------")
for i, sublist in enumerate(rouge_scores3):
    print(f"\nSummary {i+1} (Method 2):\n")
    for dictionary in sublist:
        for metric, values in dictionary.items():
            print(f"\tMetric: {metric}")
            for sub_metric, score in values.items():
                print(f"\t{sub_metric}: {score}\n")

print(f"\n##### BERT Scores #####\n")
for i, scores_tuple in enumerate(bert_scores1):
    print(f"\nExtractive Summary {i+1}:\n")
    for index, score in enumerate(scores_tuple):
        if index == 0:
            print("\tPrecision")
        elif index == 1:
            print("\tRecall")
        else:
            print("\tF1 Score")
        print(f"\tValor: {score.item()}\n")
print("-----------------------------------------")
for i, scores_tuple in enumerate(bert_scores2):
    print(f"\nSummary {i+1} (Method 1):\n")
    for index, score in enumerate(scores_tuple):
        if index == 0:
            print("\tPrecision")
        elif index == 1:
            print("\tRecall")
        else:
            print("\tF1 Score")
        print(f"\tValor: {score.item()}\n")
print("------------------------------------")
for i, scores_tuple in enumerate(bert_scores3):
    print(f"\nSummary {i+1} (Method 2):\n")
    for index, score in enumerate(scores_tuple):
        if index == 0:
            print("\tPrecision")
        elif index == 1:
            print("\tRecall")
        else:
            print("\tF1 Score")
        print(f"\tValor: {score.item()}")

# End total time
end = time.time()
print(f"\nTotal time: {end - start} seconds")
