import json
import os

from nltk import word_tokenize

import en_core_web_sm
import torch
from tqdm import tqdm
import random

PAD = "<PAD>"
UNKNOWN = "<UNKNOWN>"


class Dictionary(object):
    def __init__(self):
        self.word_to_index = {PAD: 0, UNKNOWN: 1}
        self.index_to_word = [PAD, UNKNOWN]

    def add_word(self, word):
        if word not in self.word_to_index:
            self.index_to_word.append(word)
            self.word_to_index[word] = len(self.index_to_word) - 1

        return self.word_to_index[word]

    def __len__(self):
        return len(self.index_to_word)


class Corpus(object):
    """
    Corpus class with sequence encoding functionality.

    Use 'tokenize' to both update the vocabulary as well as produce a sequence
    tensor for the document passed.

    TODO: References must be added to the dictionary.
    """

    def __init__(self, vocabulary):
        self.dictionary = Dictionary()
        self.training = []
        self.validation = []
        self.test = []
        self.vocabulary = vocabulary

        self.nlp = en_core_web_sm.load()

        for word in vocabulary:
            self.dictionary.add_word(word)

    def add_document(self, path, data="train"):
        """
        Tokenizes a Conflict JSON Wikipedia article and adds it's sequence
        tensor to the corpus.

        If a file being added does not have "title" and "sections" field, this
        function does nothing.
        :param path: The path to a training document.
        """

        parsed_json = json.load(open(path, 'r'))

        if "title" not in parsed_json or "sections" not in parsed_json:
            return

        # Collect the publication title and content sections.
        title = parsed_json["title"]
        sections = parsed_json["sections"]

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
                self.tokenize_from_text(sentence)
                sentences.append(sentence)

        if len(sentences) != 0:
            # Some documents don't have named headings.

            document_object = {
                "title": title,
                "sections": sections,
                "document": document_raw,
                "sentences": sentences
            }

            if data == "train":
                self.training.append(document_object)
            elif data == "validation":
                self.validation.append(document_object)
            else:
                self.test.append(document_object)

    def tokenize_from_text(self, text):
        """
        Given a string of text, returns a new tensor of the same length
        as words in the text containing word vectors.
        """
        text = text.replace(r'\s+', ' ')
        words = word_tokenize(text)

        # Some sections may be empty; return None in this case.
        if len(words) == 0:
            return None

        # Construct a sequence tensor for the text.
        ids = torch.LongTensor(len(words))
        for i, word in enumerate(words):
            if word in self.dictionary.word_to_index:
                ids[i] = self.dictionary.word_to_index[word]
            else:
                ids[i] = self.dictionary.word_to_index[UNKNOWN]

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

    def __init__(self, corpus, extractor, umls, data_dir,
                 batch_size=30, cuda=False, target_limit = 1):
        self.corpus = corpus
        self.extractor = extractor
        self.umls = umls
        self.training = []
        self.validation = []
        self.batch_size = batch_size
        self.cuda = cuda
        self.data_dir = data_dir
        self.target_limit = target_limit

        if len(os.listdir(data_dir)) == 0:
            self.generate_all_data()

    def generate_all_data(self):
        """
        Generates a training set for the SummaRuNNer model
        """

        # Partitions UMLS terms into 80:20 split
        num_terms = len(self.umls.terms)
        training_terms = self.umls.terms[0:int(num_terms * 0.8)]
        validation_terms = self.umls.terms[int(num_terms * 0.8):]

        print("Collecting training definitions:")
        for i, document in tqdm(enumerate(self.corpus.training)):
            for j, entity in enumerate(training_terms):
                training_ex = self.generate_one_example(document, entity)
                if training_ex is not None:
                    self.training.append(training_ex)

        print("Collecting validation definitions:")
        for i, document in tqdm(enumerate(self.corpus.validation)):
            for j, entity in enumerate(validation_terms):
                validation_ex = self.generate_one_example(document, entity)
                if validation_ex is not None:
                    self.training.append(validation_ex)

    def generate_one_example(self, document, entity):
        _, targets = self.extractor.construct_extraction_from_document(document["sentences"],
                                                                       entity["definition"])
        training_example = {
            "entity": entity["term"],
            "e_gold": entity["definition"],
            "targets": list(targets),
            "document": document
        }

        # Discards the example if it has no non-zero targets.
        if torch.sum(targets) == 0:
            return None

        # Save the data as a JSON file.
        title = document["title"].replace(" ", "_")
        training_json = os.path.join(self.data_dir, title + "_" + entity["term"] + ".json")
        with open(training_json, "w") as f:
            json.dump(training_example, f,
                      sort_keys=True,
                      indent=2)

        return training_example

    def data_loader(self, randomized=False, training=True):
        """
        Returns a new instance of the training data.

        The final batch may be have fewer than 'batch_size' documents.
        It is up to the user to decide whether to salvage or discard this batch.

        :param randomized: boolean
            Whether or not to shuffle the examples.
        :param training: boolean
            Whether or not to use the training data. If false, resorts as a
            data loader for validation.
        :return: A generator that produces 'batch_size' documents at a time.
        """

        if training:
            examples = self.corpus.training
        else:
            examples = self.corpus.validation

        if randomized:
            examples = random.shuffle(examples)

        for i in range(0, len(examples), self.batch_size):
            yield examples[i:i + self.batch_size]

    def training_loader(self, randomized=False):
        return self.data_loader(randomized=randomized)

    def development_loader(self, randomized=False):
        return self.data_loader(randomized=randomized, training=False)

    def shuffle_training(self):
        self.corpus.training = random.shuffle(self.corpus.training)

    def shuffle_validation(self):
        self.corpus.validation = random.shuffle(self.corpus.validation)

    def vectorize_sentences(self, sentences):
        """
        Given a list of sentences, returns a new list
        containing an encoding for each sentence.
        """
        return self.corpus.vectorize_sentences(sentences)