from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def fuse_sentences(
    sentences_lists,
    num_beams=1,
    num_beam_groups=1,
    num_return_sequences=1,
    diversity_penalty=0.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=2,
    temperature=1,
    max_length=256
):
    """
    Fuse sentences using a language generation model.

    Parameters:
    sentences_lists (list of lists): List of lists of sentences to fuse.
    num_beams (int): Number of beams to use in sequence search.
    num_beam_groups (int): Number of beam groups to use in sequence search.
    num_return_sequences (int): Number of sequences to generate for each input.
    diversity_penalty (float): Diversity penalty applied to the word distribution.
    repetition_penalty (float): Repetition penalty applied to the word distribution.
    no_repeat_ngram_size (int): Size of n-gram to avoid repetitions in sequence generation.
    temperature (float): Controls the entropy in sequence generation.
    max_length (int): Maximum length of the generated sequence.

    Returns:
    list of lists: List of lists of fused sentences.
    """
    
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

    fused_sentences = []

    for sentences_list in sentences_lists:
        fused_sentences_list = []

        for sentence in sentences_list:
            input_ids = tokenizer(
                f'paraphrase: {sentence}',
                return_tensors="pt", padding="longest",
                max_length=max_length,
                truncation=True,
            ).input_ids.to(device)

            outputs = model.generate(
                input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams, num_beam_groups=num_beam_groups,
                max_length=max_length, diversity_penalty=diversity_penalty
            )

            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            fused_sentences_list.append(res[0])

        fused_sentences.append(fused_sentences_list)

    return fused_sentences
