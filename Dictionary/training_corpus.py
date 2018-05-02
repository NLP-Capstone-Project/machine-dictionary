# Adapted from PyTorch examples:
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py

import json
from nltk import word_tokenize


# UMLS Initialization Code, should go in data.py, here temporarily to avoid conflicts

class UMLSCorpus(object):
    """
    UMLSCorpus class which encodes all the training examples for SummaRuNNer
    Each training example contains:
    - e: the entity from UMLS
    - e_gold: the gold standard definition from UMLS
    - t: the target representation from the extractive ROUGE tagger
    - d: the document associated with the target tag
    """

    def __init__(self, corpus, extractor, umls):
        self.corpus = corpus
        self.extractor = extractor
        self.umls = umls

    def generate_all_data(self):
        """
        Generates a training, development, and test set for the SummaRuNNer model
        """
        for i, document in enumerate(corpus.training):
            for j, entity in enumerate(umls)

    def generate_one_example(self, document, entity):
