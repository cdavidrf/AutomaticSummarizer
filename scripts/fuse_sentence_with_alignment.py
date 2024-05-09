import spacy

nlp = spacy.load("en_core_web_sm")

def fuse_sentences(sentences_list):
    """
    Fuse sentences of each sentence pair in a list of lists of sentences.

    Parameters:
    sentences (list of list of lists): List of lists of sentence pairs.

    Returns:
    list of str: List of fused sentences for each sentence pair.
    """
    
    fused_sentences = []

    for sentence_pair in sentences_list:
        fused_sentences_list = []
        
        for pair in sentence_pair:
            docs = [nlp(sentence) for sentence in pair]

            alignments = []
            doc2_to_doc1 = {}
            for token1 in docs[0]:
                for token2 in docs[1]:
                    if token1.lemma_ == token2.lemma_:
                        alignments.append((token1, token2))
                        doc2_to_doc1[token2.i] = token1.i

            fused_sentence_list = []  
            doc1_used = set()  

            for token1 in docs[0]:
                words_to_add = [token1.text]  

                for i, token2 in enumerate(docs[1]):
                    if i in doc2_to_doc1 and doc2_to_doc1[i] == token1.i:
                        if token2.text != token1.text:
                            words_to_add.append(token2.text)

                best_option = sorted(words_to_add, key=len)[0]  
                fused_sentence_list.append(best_option)
                doc1_used.add(token1.i)

            for i, token2 in enumerate(docs[1]):
                if i not in doc2_to_doc1:
                    fused_sentence_list.append(token2.text)

            fused_sentence_str = " ".join(fused_sentence_list)
            fused_sentences_list.append(fused_sentence_str)  

        fused_sentences.append(fused_sentences_list)

    return fused_sentences
