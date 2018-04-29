# Adapted from PyTorch examples:
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py

import json
from nltk import word_tokenize
import os

import torch


class Dictionary(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = []

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
    """

    def __init__(self):
        self.dictionary = Dictionary()
        self.documents = []

    def add_example(self, path):
        """
        Tokenizes a text file and adds it's sequence tensor to the corpus.
        :param path: The path to a training document.
        """
        sequence_tensor = self.tokenize(path)
        self.documents.append(sequence_tensor)

    def add_document(self, path):
        """
        Tokenizes a Semantic Scholar JSON publication and adds it's sequence
        tensor to the corpus.

        If a file being added does not have "title" and "section" field, this
        function does nothing.
        :param path: The path to a training document.
        """
        parsed_document = json.load(open(path, 'r'))

        if "title" not in parsed_document or "sections" not in parsed_document:
            return

        # Collect the publication title and content sections.
        title = parsed_document["title"]
        sections = parsed_document["sections"]

        section_tensors = []

        # Abstracts are separate from the rest of the papers, add
        # them to the section tensors for easy training.
        #
        # Not every paper will have a successfully parsed abstract.
        if "abstractText" in parsed_document:
            abstract = self.tokenize_from_text(parsed_document["abstractText"])
            section_tensors.append(abstract)

        # Vectorize every section of the paper except for references.
        exclude = ["ACKNOWLEDGEMENTS", "Authors’ Contributions"]
        for section in sections:
            if "heading" in section and section["heading"] not in exclude:
                section_tensor = self.tokenize_from_text(section["text"])

                # Handle empty section case.
                if section_tensor is not None:
                    section_tensors.append(section_tensor)

        document_object = {
            "title": title,
            "sections": section_tensors
        }
        self.documents.append(document_object)

    def tokenize_from_text(self, text):
        words = word_tokenize(text)

        # Some sections may be empty; return None in this case.
        if len(words) == 0:
            return None

        # Add the words to the dictionary.
        for word in words:
            self.dictionary.add_word(word)

        # Construct a sequence tensor for the text.
        ids = torch.LongTensor(len(words))
        for i, word in enumerate(words):
            ids[i] = self.dictionary.word_to_index[word]

        return ids

    def tokenize_from_path(self, path):
        """
        Tokenize a text file into a sequence tensor.
        :param path: The path to a training document.
        :return A sequence tensor of the document of dimensions
            (Length of document,) s.t. the ith column is the integer
            representation of the ith word in the document.

            Indices are consistent with all other documents used for this
            corpus.
        """
        assert(os.path.exists(path))

        lines = []
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = word_tokenize(line) + ['<EOS>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                lines.append(words)

        # Convert the document into its own sequence tensor.
        ids = torch.LongTensor(tokens)
        tokens = 0
        for line in lines:
            for word in line:
                ids[tokens] = self.dictionary.word_to_index[word]
                tokens += 1

        return ids
