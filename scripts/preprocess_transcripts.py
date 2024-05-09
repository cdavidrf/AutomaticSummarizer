import nltk
import re
from nltk.tokenize import sent_tokenize

def preprocess(transcripts):
    """
    Preprocesses the transcripts by removing brackets, remarks, and other elements,
    tokenizes the text into sentences, and performs other cleaning operations.

    Parameters:
    transcripts (list of str): List of transcripts to preprocess.

    Returns:
    list of list of str: A list of lists where each sublist contains the preprocessed sentences of a transcript.
    """
    
    nltk.download('punkt')

    preprocessed_sentences = []

    for transcript in transcripts:
        text_without_brackets = re.sub(r'\[.*?\]', '', transcript)
        text_without_remarks = re.sub(r'Prepared Remarks:', '', text_without_brackets)
        lines = text_without_remarks.split('\n')

        for i, line in enumerate(lines):
            if "Questions and Answers:" in line:
                lines = lines[:i]
                break

        final_text = '\n'.join(lines)

        sentences = sent_tokenize(final_text)
        sentences = [sentence.lower() for sentence in sentences]
        sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
        pattern = r'[\n\r]'
        sentences = [re.sub(pattern, ' ', sentence) for sentence in sentences]

        preprocessed_sentences.append(sentences)

    return preprocessed_sentences
