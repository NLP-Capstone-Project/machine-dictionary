# Adapted from PyTorch examples:
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py

import json
from nltk import word_tokenize

from allennlp.nn.util import sort_batch_by_length
import en_core_web_sm
import spacy
import torch
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

        section_tensors = []

        # Vectorize every section of the paper except for references.
        exclude = ["References"]
        document_raw = ""
        for section in sections:
            if "heading" in section and section["heading"] not in exclude:
                section_tensor = self.tokenize_from_text(section["text"])

                # Handle empty section case.
                if section_tensor is not None:
                    section_tensors.append(section_tensor)

                document_raw += ("\n" + section["text"])

        # Collect the entire document along with its sentences.
        document = self.tokenize_from_text(document_raw)
        parsed_document = self.nlp(document_raw)
        sentences = []
        for s in parsed_document.sents:
            sentence = str(s).strip()

            # Discard sentences that are less than 3 words long.
            if len(sentence.split()) > 3:
                sentences.append(self.tokenize_from_text(sentence))

        if len(sentences) != 0:
            # Some documents don't have named headings.

            document_object = {
                "title": title,
                "sections": section_tensors,
                "document": document,
                "sentences": sentences
            }

            if data == "train":
                self.training.append(document_object)
            elif data == "validation":
                self.validation.append(document_object)
            else:
                self.test.append(document_object)

    def tokenize_from_text(self, text):
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


class SemanticScholarDataset(object):

    def __init__(self, corpus, batch_size=30, cuda=False):
        """

        :param corpus: The corpus of documents containing
        :param batch_size:
        """
        self.corpus = corpus
        self.batch_size = batch_size
        self.cuda = cuda

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

    @staticmethod
    def sentence_loader(batch, cuda=False):
        """
        Given a batch of documents, creates a generator that produces
        sentences and the labels for each of them.

        :param batch: A batch of documents of the form returned by train_loader.
        :param cuda: Boolean as to whether or not to use the GPU.
        :return: A generator that produces stacked sentences and labels.
        """
        document_lengths = [len(doc["sentences"]) for doc in batch]
        max_document_length = document_lengths[0]
        for i in range(max_document_length):
            # Create a stacked sentence tensor (unsorted).
            # It will be up to the model's forward function to properly
            # sort and restore.
            sentences = [x["sentences"][i] for x in batch]
            max_length = len(max(sentences, key=lambda x: len(x)))

            # Represents the stacked sentences for this iteration.
            # Shape: (batch size x max sentence length)
            sentences_tensor = torch.zeros(len(sentences), max_length).long()

            for j, sentence in enumerate(sentences):
                sentences_tensor[j, :len(sentence)] = sentence

            if cuda:
                sentences_tensor = sentences_tensor.cuda()

            yield sentences_tensor

    @staticmethod
    def mask_by_batch(stacked_sentences, cuda=False):
        """
        Given stacked sentences returned by a sentence_loader,
        returns a new vector of 'batch_size' length that is 1 everywhere
        there is a valid sentence in batch and 0 elsewhere.

        When computing loss, multiply predictions by this mask to prevent
        learning from padding.
        """
        sentence_lengths = stacked_sentences.sum(dim=1)
        sentence_mask = (sentence_lengths != 0).type(
            torch.LongTensor if cuda else
            torch.cuda.LongTensor
        )

        return sentence_mask

    def training_loader(self, randomized=False):
        return self.data_loader(randomized=randomized)

    def development_loader(self, randomized=False):
        return self.data_loader(randomized=randomized, training=False)

    def shuffle_training(self):
        self.corpus.training = random.shuffle(self.corpus.training)

    def shuffle_validation(self):
        self.corpus.validation = random.shuffle(self.corpus.validation)
