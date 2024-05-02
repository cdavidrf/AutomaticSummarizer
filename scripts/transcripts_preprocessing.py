import nltk
import re
from nltk.tokenize import sent_tokenize

def preprocess_transcripts(transcripts):
    """
    Preprocesa los transcripts eliminando corchetes, remarks, y otros elementos,
    tokeniza el texto en oraciones, y realiza otras operaciones de limpieza.

    Parameters:
    transcripts (list of str): Lista de transcripts a preprocesar.

    Returns:
    list of list of str: Una lista de listas donde cada sublista contiene las oraciones preprocesadas de un transcript.
    """
    
    print("\nPreprocessing transcripts...\n")

    # Descargar el tokenizer de oraciones de NLTK
    nltk.download('punkt')

    # Lista para almacenar todas las listas de oraciones
    total_sentences = []

    # Iterar sobre cada transcript en transcripts
    for transcript in transcripts:
        # Utilizamos expresiones regulares para preprocesar el texto
        texto_sin_corchetes = re.sub(r'\[.*?\]', '', transcript)  # Eliminar corchetes y su contenido
        texto_sin_remarks = re.sub(r'Prepared Remarks:', '', texto_sin_corchetes)  # Eliminar "Prepared Remarks:"

        # Dividir el texto en líneas
        lineas = texto_sin_remarks.split('\n')

        # Eliminar todo después de "Questions and Answers:"
        for i, linea in enumerate(lineas):
            if "Questions and Answers:" in linea:
                lineas = lineas[:i]
                break

        # Unir las líneas nuevamente en un solo texto
        texto_final = '\n'.join(lineas)

        # Tokenizar el texto en oraciones
        sentences = sent_tokenize(texto_final)

        # Convertir todas las oraciones a minúsculas
        sentences = [sentence.lower() for sentence in sentences]

        # Eliminar signos de puntuación
        sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]

        # Eliminar caracteres de nueva línea o retorno
        patron = r'[\n\r]'
        sentences = [re.sub(patron, ' ', sentence) for sentence in sentences]

        # Agregar la lista de oraciones a total_sentences
        total_sentences.append(sentences)

    return total_sentences





