import json
import os
import re

import en_core_web_sm
from nltk.tokenize import word_tokenize
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
        parsed_document = self.nlp(bytes(document_raw, 'utf-8', 'ignore')
                                   .decode('utf-8'))

        # Add tokens to the dictionary.
        self.tokenize_from_text(parsed_document.text)

        sentences = []
        for sent in parsed_document.sents:
            # Discard sentences that are less than 3 words long.
            if len(sent) > 3:
                # Precautionary vectorization.
                sentence = ' '.join(sent.text.split())  # Remove excess whitespace.
                sentences.append(sentence)

        if len(sentences) != 0:
            # Some documents don't have named headings.
            document_object = {
                "title": title,
                "sentences": sentences
            }

            return document_object
        else:
            return None

    def save_processed_document(self, src, dst):
        document_object = self.process_document(src)
        if document_object is not None:
            with open(dst, "w", encoding='utf-8') as f:
                json.dump(document_object, f,
                          ensure_ascii=False,
                          sort_keys=True,
                          indent=2)

    def tokenize_from_text(self, words):
        """
        Given a list of words, returns a new tensor of the same length
        as words in the text containing word vectors.
        """
        words = word_tokenize(bytes(words, 'utf-8', 'replace').decode('utf-8'))

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

    def __init__(self, dictionary, extractor, umls, cuda=False, target_limit=1):
        self.dictionary = dictionary
        self.extractor = extractor
        self.umls = umls
        self.training = []
        self.validation = []
        self.cuda = cuda
        self.target_limit = target_limit

        # For easy parsing of unicode
        self.nlp = en_core_web_sm.load()

    def collect_all_data(self, data_dir):
        if data_dir is not None and os.path.exists(data_dir):
            for example in os.listdir(data_dir):
                example_path = os.path.join(data_dir, example)
                example_json = json.load(open(example_path, 'r'))
                self.training.append(example_json)

    def generate_all_data(self, bio_dir, parsed_dir):
        """
        Given the directory to all of the parsed Semantic Scholar data,
        compares each document to every entity in UMLS and produces a
        triplet iff the term is contained in the document with at least one
        sentence labeled.
        """
        for i, document_name in enumerate(os.listdir(parsed_dir)):
            document_path = os.path.join(parsed_dir, document_name)
            document_json = json.load(open(document_path, 'r'))
            for definition, terms in self.umls.definition_mappings.items():
                training_ex = self.generate_example(document_json, definition, terms,
                                                    bio_dir)
                if training_ex:
                    self.training += training_ex

    def generate_example(self, document_json, definition, terms, bio_dir):
        """
        Generates a series of training examples built on a document, a definition,
        and all the terms that share that definition.

        If an entity is not mentioned in the document, don't attempt to extract
        a definition for it.

        If a reference is less than 3 words, don't attempt extraction.
        """

        # Remove HTML chars from definition if present.
        definition = re.sub(r'<[^<>]+>', " ", definition)

        if len(definition.split()) < 3:
            return None

        sentences = document_json["sentences"]
        document_raw = ' '.join(document_json["sentences"])
        found = False
        for term in terms:
            if term in document_raw:
                found = True
                break

        if not found:
            return None

        print("\nPAPER:", document_json["title"], "TERMs:", terms)
        print("TERM FOUND:", found)

        # Uncomment to toggle:
        # Cosine similarity
        # extracted, targets = self.extractor.cosine_similarity(sentences, definition)

        # Skip grams
        # extracted, targets = self.extractor.skipgram_similarity(self.extractor
        #                                                         .construct_skipgram_map(sentences),
        #                                                         sentences,
        #                                                         definition)
        # Both
        extracted, targets = self.extractor.experimental_similarity(sentences, definition)

        # Discards the example if it has no non-zero targets.
        if torch.sum(targets) == 0:
            return None

        training_examples = []
        for term in terms:
            import pdb
            training_example = {
                "entity": term,
                "e_gold": definition,
                "extracted": extracted,
                "targets": list(targets),
                "document": document_json
            }

            # Save the data as a JSON file (first five words).
            #
            # Some titles contain strange characters; enforce alphanumerics
            # for easy file creation.
            title = document_json["title"]
            title = re.sub(r'[^a-zA-Z0-9]', '_', title)
            title = '_'.join(title.split()[:5])
            term = re.sub(r'[^a-zA-Z0-9]', '_', term)
            training_file = title + "_" + term + ".json"
            training_json = os.path.join(bio_dir, training_file)

            with open(training_json, "w") as f:
                json.dump(training_example, f,
                          sort_keys=True,
                          ensure_ascii=False,
                          indent=2)

            training_examples.append(training_example)

        return training_examples

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