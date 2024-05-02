# Autor: 
import pandas as pd
import time

# Importar los scripts necesarios
import scripts.transcripts_preprocessing as transcripts_preprocessing
import scripts.vectorize_with_tfidf as vectorize_with_tfidf
import scripts.vectorize_with_cohere as vectorize_with_cohere
import scripts.vectorize_with_jinaai as vectorize_with_jinaai
import scripts.calculate_cosine_matrix as calculate_cosine_matrix
import scripts.find_threshold as find_threshold
import scripts.create_graph as create_graph
import scripts.find_connected_components as find_connected_components
import scripts.select_sentences_with_mmr as select_sentences_with_mmr
import scripts.sentence_fusion as sentence_fusion
import scripts.calculate_rouge as calculate_rouge
import scripts.calculate_bertscore as calculate_bertscore

# Inicializar el temporizador
start = time.time()
# Leer el archivo CSV
#data = pd.read_csv("data/resumen_individual.csv",sep=",")
data = pd.read_csv("data/resumenes_4muestras.csv",sep=",")

# Mostrar información sobre el dataset
print(f"Data sample:\n{data.head()}\n")
print(f"Data shape: {data.shape}")

# Seleccionamos la columna 'transcript'
transcripts = data['text']

# Preprocesamiento de los textos
preprocessed_transcripts = transcripts_preprocessing.preprocess_transcripts(transcripts)

# Crear vectores TF-IDF
tfidf_vectors = vectorize_with_tfidf.calculate_tfidf_vectors(preprocessed_transcripts)

# Crear vectores Cohere
#cohere_vectors = vectorize_with_cohere.calculate_cohere_vectors(preprocessed_transcripts)

# Crear vectores JinaAI
#jinaai_vectors = vectorize_with_jinaai.calculate_jina_ai_vectors(preprocessed_transcripts)

# Calcular matrices de similitud de coseno
cosine_similarities = calculate_cosine_matrix.cosine_similarity_matrix(tfidf_vectors)

# Calcular el umbral de similitud
optimal_threshold, max_suma_total = find_threshold.find_optimal_threshold(cosine_similarities, preprocessed_transcripts)

# Crear grafos de oraciones
graph_list = create_graph.create_sentence_graph(cosine_similarities, optimal_threshold, preprocessed_transcripts)

# Encontrar componentes conectados
cc_list,_ = find_connected_components.connected_components(graph_list)

# Seleccionar oraciones representativas utilizando MMR
summary_list_tfidf, selected_sentences = select_sentences_with_mmr.select_sentences(cc_list)

#print(f"Selected Sentences:\n{selected_sentences}\n")

# Fusionar oraciones seleccionadas en un solo string
fused_sentences = sentence_fusion.fuse_sentences(selected_sentences)

# Concatenar las oraciones fusionadas para cada texto y obtener el resumen abstractivo
abstractive_summaries=[]
for i, modified_sentences in enumerate(fused_sentences):
  abstractive_summary = '\n'.join(modified_sentences)
  abstractive_summaries.append(abstractive_summary)
  print(f"Abstractive Summary {i+1}:\n\n{abstractive_summary}\n")

# Calcular ROUGE scores
reference_summaries=data['reference_summary']
rouge_scores = calculate_rouge.calculate_rouge_scores(abstractive_summaries, reference_summaries)

# Calcular BERTScore
bert_scores = calculate_bertscore.calculate_bert_scores(abstractive_summaries, reference_summaries)

# Finalizar el temporizador
end = time.time()
print(f"Total Time taken: {end - start} seconds")
