import json
import os

import en_core_web_sm
import torch
import random

PAD = "<PAD>"
UNKNOWN = "<UNKNOWN>"


class Dictionary(object):
    def __init__(self, vocabulary):
        self.word_to_index = {PAD: 0, UNKNOWN: 1}
        self.index_to_word = [PAD, UNKNOWN]
        self.vocabulary = vocabulary

        self.training = []
        self.validation = []
        self.test = []

        self.nlp = en_core_web_sm.load()

        for word in vocabulary:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_index:
            self.index_to_word.append(word)
            self.word_to_index[word] = len(self.index_to_word) - 1

        return self.word_to_index[word]

    def __len__(self):
        return len(self.index_to_word)

    def process_document(self, path):
        """
        Tokenizes a Conflict JSON Wikipedia article and adds it's sequence
        tensor to the corpus.

        Returns none if a file being added does not have "title" and "sections" field.
        :param path: The path to a training document.
        """

        parsed_json = json.load(open(path, 'r'))

        if "metadata" not in parsed_json:
            return None

        # Collect the publication title and content sections.
        parsed_json = parsed_json["metadata"]
        title = parsed_json["title"]
        sections = parsed_json["sections"]

        if not title or not sections:
            return None

        # Construct the entire document from its sections.
        # Vectorize every section of the paper except for references.
        exclude = ["References", "Acknowledgments", "Appendix"]
        document_raw = ""
        for section in sections:
            # Collect only semantically significant sections.
            if "heading" in section and section["heading"] not in exclude:
                document_raw += ("\n" + section["text"])

        # Vectorize all words in the document.
        parsed_document = self.nlp(document_raw)
        sentences = []
        for s in parsed_document.sents:
            sentence = str(s).strip()

            # Discard sentences that are less than 3 words long.
            if len(sentence.split()) > 3:
                # Precautionary vectorization.
                sentence = list(self.nlp(sentence))
                sentence = [word.text for word in sentence]
                self.tokenize_from_text(sentence)
                sentences.append(sentence)

        if len(sentences) != 0:
            # Some documents don't have named headings.

            document_object = {
                "title": title,
                "sections": sections,
                "sentences": sentences
            }

            return document_object
        else:
            return None

    def save_processed_document(self, src, dst):
        document_object = self.process_document(src)
        if document_object is not None:
            with open(dst, "w") as f:
                json.dump(document_object, f,
                          sort_keys=True,
                          indent=2)

    def tokenize_from_text(self, words):
        """
        Given a string of text, returns a new tensor of the same length
        as words in the text containing word vectors.

        Also adds every word in the string of test to the dictionary.
        """

        # Some sections may be empty; return None in this case.
        if len(words) == 0:
            return None

        # Construct a sequence tensor for the text.
        ids = torch.LongTensor(len(words))
        for i, word in enumerate(words):
            if word in self.word_to_index:
                ids[i] = self.word_to_index[word]
            else:
                ids[i] = self.word_to_index[UNKNOWN]

        return ids

    def vectorize_sentences(self, sentences):
        """
        Given a list of sentences, returns a new list
        containing an encoding for each sentence.
        """
        return [self.tokenize_from_text(sent) for sent in sentences]


class UMLSCorpus(object):
    """
    UMLSCorpus class which encodes all the training examples for SummaRuNNer
    Each training example contains:
    - e: the entity from UMLS
    - e_gold: the gold standard definition from UMLS
    - t: the target representation from the extractive ROUGE tagger
    - d: the document associated with the target tag
    """

    def __init__(self, dictionary, extractor, umls, data_dir, parsed_dir,
                 cuda=False, target_limit=1):
        self.dictionary = dictionary
        self.extractor = extractor
        self.umls = umls
        self.training = []
        self.validation = []
        self.cuda = cuda
        self.data_dir = data_dir
        self.parsed_dir = parsed_dir
        self.target_limit = target_limit

        # For easy parsing of unicode
        self.nlp = en_core_web_sm.load()

        # If the data directory is not empty, collect the training
        # examples.
        if data_dir is not None and os.path.exists(data_dir):
            for example in os.listdir(data_dir):
                example_path = os.path.join(data_dir, example)
                example_json = json.load(open(example_path, 'r'))
                self.training.append(example_json)

    def generate_all_data(self):
        """
        Given the directory to all of the parsed Semantic Scholar data,
        compares each document to every entity in UMLS and produces a
        triplet iff the term is contained in the document with at least one
        sentence labeled.
        """
        for i, document_name in enumerate(os.listdir(self.parsed_dir)):
            document_path = os.path.join(self.parsed_dir, document_name)
            document_json = json.load(open(document_path, 'r'))
            for j, entity in enumerate(self.umls.terms):
                sentence_nlps = [self.nlp(sent) for sent in document_json["sentences"]]
                training_ex = self.generate_one_example(document_json, sentence_nlps,
                                                        entity)
                if training_ex:
                    self.training.append(training_ex)

    def generate_one_example(self, document_json, sentence_nlps, entity):
        """
        Generates a single training example.

        If an entity is not mentioned in the document, don't attempt to extract
        a definition for it.
        """

        if entity["term"] not in '\n'.join(document_json["sentences"]):
            return None

        targets = self.extractor.extraction_cosine_similarity(sentence_nlps,
                                                              entity["definition"])

        # Discards the example if it has no non-zero targets.
        if torch.sum(targets) == 0:
            return None

        training_example = {
            "entity": entity["term"],
            "e_gold": entity["definition"],
            "targets": list(targets),
            "document": document_json
        }

        # Save the data as a JSON file (first five words).
        title = '_'.join(document_json["title"].split()[:5])
        training_json = os.path.join(self.data_dir, title + "_" + entity["term"].replace(" ", "_") + ".json")
        with open(training_json, "w") as f:
            json.dump(training_example, f,
                      sort_keys=True,
                      indent=2)

        return training_example

    def data_loader(self, batch_size, randomized=False, training=True):
        """
        Returns a new instance of the training data.

        The final batch may be have fewer than 'batch_size' documents.
        It is up to the user to decide whether to salvage or discard this batch.
        :param batch_size: int
            Interval of partitioning across the dataset.
        :param randomized: boolean
            Whether or not to shuffle the examples.
        :param training: boolean
            Whether or not to use the training data. If false, resorts as a
            data loader for validation.
        :return: A generator that produces 'batch_size' documents at a time.
        """

        if training:
            examples = self.training
        else:
            examples = self.validation

        if randomized:
            examples = random.shuffle(examples)

        for i in range(0, len(examples), batch_size):
            yield examples[i:i + batch_size]

    def training_loader(self, batch_size, randomized=False):
        return self.data_loader(batch_size, randomized=randomized)

    def development_loader(self, batch_size, randomized=False):
        return self.data_loader(batch_size, randomized=randomized, training=False)

    def shuffle_training(self):
        self.training = random.shuffle(self.training)

    def shuffle_validation(self):
        self.validation = random.shuffle(self.validation)

    def vectorize_sentences(self, sentences):
        """
        Given a list of sentences, returns a new list
        containing an encoding for each sentence.
        """
        return self.dictionary.vectorize_sentences(sentences)