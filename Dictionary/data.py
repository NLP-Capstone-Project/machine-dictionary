# Adapted from PyTorch examples:
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py

import json
from nltk import word_tokenize

import en_core_web_sm
import spacy
import torch

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
