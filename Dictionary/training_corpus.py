# UMLS Initialization Code, should go in data.py, here temporarily to avoid conflicts
definition_path = "" # to be populated from arguments
synonym_path = "" # to be populated from arguments
umls = UMLS(definition_path, synonym_path)
umls.generate_all_definitions()

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
        self.training = []

    def generate_all_data(self):
        """
        Generates a training set for the SummaRuNNer model
        """
        for i, document in enumerate(corpus.training):
            for j, entity in enumerate(umls.terms):
                self.generate_one_example(document, entity)

    def generate_one_example(self, document, entity):
        training_example = {
            entity: entity["term"],
            e_gold: entity["definition"],
            target: self.extractor.construct_extraction_from_document(document["sentences"], entity["definition"]),
            document: document
        }
        self.training.append(training_example)
