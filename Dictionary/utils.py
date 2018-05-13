import json
import torch

from nltk.tokenize import word_tokenize


def word_vector_from_seq(sequence_tensor, i):
    """
    Collect the vector representation of the ith word in a sequence tensor.
    :param sequence_tensor: The document in which to collect from.
    :param i: The ith index of the document.
    :return: A `torch.LongTensor()` containing the document's ith word's
        encoding relative to this corpus.
    """

    word = torch.LongTensor(1)
    word[0] = sequence_tensor[i]
    return word


def extract_tokens_from_json(path, nlp):
    """
    Tokenizes a Conflict JSON Wikipedia article and returns a list
    of its tokens..

    If a file does not have "title" and "sections" field, this
    function returns an empty list.
    :param path: The path to a training document.
    """
    parsed_document = json.load(open(path, 'r'))

    if "metadata" not in parsed_document:
        return []

    # Collect the content sections.
    parsed_document = parsed_document["metadata"]
    sections = parsed_document["sections"]
    if not sections:
        return []

    tokens = []

    # Collect tokens from every section of the paper except for references.
    exclude = ["References"]
    for section in sections:
        if "heading" in section and section["heading"] not in exclude:
            words = word_tokenize(bytes(section["text"], 'utf-8', 'ignore').decode('utf-8'))
            tokens += words

    return tokens
