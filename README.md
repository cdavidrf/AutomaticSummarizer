# Automatic Text Summarization Project

This project aims to achieve automatic text summarization using graph-based techniques. It implements a workflow that includes text vectorization, similarity calculation, graph creation, connected components computation, representative sentence selection, and sentence fusion to generate summaries. Additionally, evaluation metrics such as ROUGE and BERTScore are calculated to assess the quality of the generated summaries.


## Environment Requirements

It is recommended to use Python 3.10.12 to run this project. 

To install Python, you can follow the instructions provided in the [official Python documentation](https://docs.python.org/3/).

Note: To use Cohere for text vectorization, you will need to obtain an API key from Cohere by accessing the following link [Cohere API Keys](https://dashboard.cohere.com/api-keys). Once you have obtained the API key, please set it up in your environment. Create a file named `.env` in the root directory of the project and add the following line with your API key:

```
COHERE_API_KEY = "your_api_key_here"
```

## Project Execution

To run the project, follow these steps:

1. Install project dependencies by running `pip install -r requirements.txt`.
   
```
pip install -r requirements.txt
```

2. Download the English language model for SpaCy by running the following command: 

```
python3 -m spacy download en_core_web_sm
```

3. Execute the `main.py` script to generate text summaries. Make sure to have the necessary data files in the `data/` folder.

```
python main.py
```

The execution process includes the following stages:

- Transcripts Preprocessing: Text transcripts are processed to clean and prepare the data.
- Text Vectorization: Techniques like TF-IDF, Cohere, and Jina AI are used to convert text into numerical vectors.
- Similarity Calculation: The similarity between text vectors is calculated using metrics like cosine similarity.
- Graph Creation: Graphs are created using similarity matrices to represent the text structure.
- Connected Components Computation: Connected components are identified within the graph structure to capture meaningful clusters of sentences.
- Representative Sentence Selection: The MMR technique is used to select the most relevant sentences for the final summary.
- Sentence Fusion: Selected sentences are fused to generate summaries.
- Results Evaluation: Evaluation metrics such as ROUGE and BERTScore are calculated to assess the quality of the generated summaries.

## References
* [NLTK](https://www.nltk.org/)
* [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [Cohere Embeddings](https://docs.cohere.com/docs/embeddings)
* [Jina AI Embeddings](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
* [Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
* [NetworkX](https://github.com/networkx/networkx)
* [MMR](https://medium.com/tech-that-works/maximal-marginal-relevance-to-rerank-results-in-unsupervised-keyphrase-extraction-22d95015c7c5)
* [Humarin Paraphraser](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)
* [BERTScore](https://github.com/Tiiiger/bert_score)
* [Rouge](https://pypi.org/project/rouge/)
* [SpaCy](https://spacy.io/)






