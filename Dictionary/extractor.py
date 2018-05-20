import copy

import spacy
import nltk
from nltk.util import ngrams
from pythonrouge.pythonrouge import Pythonrouge
import torch
from tqdm import tqdm
# from .glove import load_embeddings
import torch.nn as nn

import logging
import mmap

from allennlp.data import Vocabulary
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)



class Extractor(object):
    """
    Extractor class with ROUGE evaluation functionality.

    Example:

    summary = [[" Tokyo is the capital of Japan"], ["Tokyo is the commerce center of Japan"], ["I like puppies"]]
    reference = ["The capital of Japan, Tokyo, is the center of Japanese economy."]

    ext = Extractor(0.05, 'ROUGE-2')

    ext.extraction_rouge(summary, reference) will give us [Tokyo is the capital of Japan]

    Example execution of how a new metric might be faster. Uncomment to run

    ext = Extractor(0.05, 'ROUGE-2')

    summary = [" Tokyo is the capital of Japan", "Tokyo is the commerce center of Japan", "I like puppies"]
    reference = "The capital of Japan, Tokyo, is the center of Japanese economy."

    summaryROUGE = [[" Tokyo is the capital of Japan"], ["Tokyo is the commerce center of Japan"], ["I like puppies"]]
    referenceROUGE = ["The capital of Japan, Tokyo, is the center of Japanese economy."]

    ext.extraction_ngram(summary, reference)
    ext.extraction_rouge(summaryROUGE, referenceROUGE)

    """

    def __init__(self, threshold, rouge_type='ROUGE-1', n_gram=2, to_lowercase=True):
        self.threshold = threshold
        self.rouge_type = rouge_type
        self.n_gram = n_gram
        self.to_lowercase = to_lowercase
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.obtain_stopwords("../stopwords.txt")
        glove_path = "../glove/glove.6B.50d.txt"
        vocab_path = "../vocabulary_20.txt"
        self.embedding_matrix, self.vocab = load_embeddings(glove_path, vocab_path)
        self.embedding_dim = self.embedding_matrix.size(1)


    def obtain_stopwords(self, filename):
        file = open(filename, "r")
        return file.read().splitlines()

    def strip_stopwords(self, sentence):
        sentence_split = sentence.split()
        if self.to_lowercase:
            sentence_split = [word.lower() for word in sentence_split]
            result_words = [word for word in sentence_split if word.lower() not in self.stopwords]
        else:
            result_words = [word for word in sentence_split if word not in self.stopwords]

        return ' '.join(result_words)

    def word_vector_similarity(self, document_sentences, reference):
        return None

    def calculate_word_vector_similarity(self, sentence, reference):
        averaged_sentence = self.average_word_vectors(sentence)
        averaged_reference = self.average_word_vectors(reference)

    def average_word_vectors(self, sentence):
        words = sentence.split(' ')
        embeddings = torch.FloatTensor(len(words), self.embedding_dim)
        for i, word in enumerate(words):
            index = self.vocab.get_token_index(word)
            embedded_word = self.embedding_matrix[index]
            embeddings[i] = embedded_word
            print(embedded_word)
        return torch.mean(embeddings, 0)

    def extraction_rouge(self, document_sentences, reference):
        """
        Uses a greedy approach to find the sentences which maximize the ROUGE score
        with respect to the reference definition.
        """
        extracted = []
        ret_tensor = torch.zeros(len(document_sentences)).long()
        score = 0
        reference = [[[reference]]]
        print("Generating training example:")
        for i, sentence in tqdm(enumerate(document_sentences)):
            extracted.append([sentence])
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=extracted, reference=reference,
                                n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                                recall_only=True, stemming=True, stopwords=True,
                                word_level=True, length_limit=True, length=50,
                                use_cf=False, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
            temp_score = rouge.calc_score()
            if (temp_score[self.rouge_type]) > score:
                score = temp_score[self.rouge_type]
                ret_tensor[i] = 1
            else:
                extracted = extracted[:len(extracted) - 1]

        return ret_tensor

    def cosine_similarity(self, sentences, reference, threshold=0.93):
        """
        Collect sentences one at a time using cosine similarity as the heuristic.

        Each sentence is compared against the reference to determine if it should
        be selected.
        :param sentences: List of list of words representing the document.
        :param reference: The reference to the UMLS term to define.
        :param threshold: Minimum cosine similarity score in order to be included.
        :return: A tensor where 1 means a sentence should be included.
        """
        reference = self.strip_stopwords(reference)
        reference = self.nlp(reference)
        ret_tensor = torch.zeros(len(sentences)).long()
        scores = []

        for i, sentence in tqdm(list(enumerate(sentences))):
            cosine_similarity = reference.similarity(self.nlp(sentence))
            score = (i, sentence, cosine_similarity)
            scores.append(score)

        extracted_sentences = []
        for index, sentence, score in scores:
            if score > threshold:
                extracted_sentences.append(sentence)
                ret_tensor[index] = 1

        best_match = max(scores, key=lambda x: x[-1])
        print("REFERENCE:", reference.text)
        print("MAX SCORE:", best_match[-1])
        print("MAX SCORING SENTENCE:", best_match[1])
        return extracted_sentences, ret_tensor

    def greedy_cosine_similarity(self, sentences, reference,
                                 threshold=0.5, delta=0.1):
        """
        Collect sentences greedily using cosine similarity as the heuristic.
        :param sentences: List of list of words representing the document.
        :param reference: The reference to the UMLS term to define.
        :param threshold: Minimum cosine similarity score in order to be included.
        :return: A tensor where 1 means a sentence should be included.
        """
        reference = self.strip_stopwords(reference)
        reference = self.nlp(reference)
        ret_tensor = torch.zeros(len(sentences)).long()
        score = 0.0
        extracted = []
        for i in tqdm(range(len(sentences))):
            extracted.append(i)  # consider the sentence
            summary = ""  # generate a running summary including the current sentence
            for sentence in extracted:
                summary += self.strip_stopwords(sentences[sentence]) + " "
            cosine_similarity = reference.similarity(self.nlp(summary))
            if cosine_similarity >= threshold and cosine_similarity - score > delta:
                ret_tensor[i] = 1
                score = cosine_similarity
            else:
                extracted = extracted[:len(extracted) - 1]
        return [self.strip_stopwords(sentences[e]) for e in extracted], ret_tensor

    def experimental_similarity(self, sentences, reference,
                                skip_threshold=15, cosine_threshold=0.90):
        """
        Combines skip grams and cosine similarity for a more thorough check.
        :param sentences: List of list of words representing the document.
        :param reference: The reference to the UMLS term to define.
        :param skip_threshold: Minimum number of reference skipgrams
            that must intersect with reference skipgrams to be included.
        :param cosine_threshold: Minimum cosine similarity score
            in order to be included.
        :return: The list of extracted sentences and a tensor
            where 1 means a sentence should be included.
        Current configuration:
            - skip-bigrams:
                - all to lowercase
                - greedy
            - cosine similarity
                - all to lowercase
                - removes stopwords
        """
        skipgram_map = self.construct_skipgram_map(sentences)
        reference_skipgrams = self.construct_skipgram_set_from_sentence(reference)
        reference_strip = self.strip_stopwords(reference)
        reference_nlp = self.nlp(reference_strip)
        extracted = []
        ret_tensor = torch.zeros(len(skipgram_map)).long()

        if len(reference_skipgrams) == 0:
            import pdb
            pdb.set_trace()

        max_num_skipgrams = 0
        max_cosine_distance = 0
        for i in tqdm(range(len(skipgram_map))):
            cosine_distance = 0
            extracted.append(i)
            intersection = copy.deepcopy(reference_skipgrams)
            for index in extracted:
                intersection &= skipgram_map[index]

            num_hit_skipgrams = len(intersection)
            keep = False
            if num_hit_skipgrams > skip_threshold:
                sentence_nlp = self.nlp(self.strip_stopwords(sentences[i]))
                cosine_distance = reference_nlp.similarity(sentence_nlp)
                if cosine_distance > cosine_threshold:
                    ret_tensor[i] = 1
                    keep = True

            if not keep:
                extracted = extracted[:len(extracted) - 1]

            max_num_skipgrams = max(max_num_skipgrams, num_hit_skipgrams)
            max_cosine_distance = max(cosine_distance, max_cosine_distance)

        extracted_sentences = [sentences[idx] for idx in extracted]
        print("REFERENCE:", reference)
        print("CHOSEN:", extracted_sentences)
        print("MAX SKIPGRAM MATCHES:", max_num_skipgrams)
        print("MAX COSINE SIMILARITY:", max_cosine_distance)
        return extracted_sentences, ret_tensor

    def skipgram_similarity(self, skipgram_map, sentences, reference,
                            threshold=0.3):
        reference_skipgrams = self.construct_skipgram_set_from_sentence(reference)
        extracted = []
        ret_tensor = torch.zeros(len(skipgram_map)).long()

        ratio = 0
        for i in tqdm(range(len(skipgram_map))):
            extracted.append(i)
            intersection = copy.deepcopy(reference_skipgrams)
            for index in extracted:
                intersection &= skipgram_map[index]

            likeness = len(intersection) / len(reference_skipgrams)
            if likeness > threshold:
                ret_tensor[i] = 1
            else:
                extracted = extracted[:len(extracted) - 1]

            ratio = max(ratio, likeness)

        extracted_sentences = [sentences[idx] for idx in extracted]
        print("REFERENCE:", reference)
        print("CHOSEN:", extracted_sentences)
        print("MAX RATIO:", ratio)
        return extracted_sentences, ret_tensor

    def construct_skipgram_map(self, sentences):
        skipgram_map = {}
        for i, sentence in enumerate(sentences):
            skipgram_map[i] = self.construct_skipgram_set_from_sentence(sentence)
        return skipgram_map

    def construct_skipgram_set_from_sentence(self, sentence):
        skipgrams = set()
        words = sentence.split(' ')
        sentence_len = len(words)
        if self.to_lowercase:
            words = [word.lower() for word in words]
        for j in range(sentence_len):
            for k in range(j + 1, sentence_len):
                skipgrams.add((words[j], words[k]))
        return skipgrams





def read_vocabulary(vocab_path):
    train_vocab = Vocabulary()
    vocab_file = open(vocab_path, "r")
    vocab = set([word.strip() for word in vocab_file.readlines()])
    for word in vocab:
        train_vocab.add_token_to_namespace(word)
    train_vocab.add_token_to_namespace("@@@UNKNOWN@@@")
    return train_vocab


def load_embeddings(glove_path, vocab_path):
    """
    Create an embedding matrix for a Vocabulary.
    Usage: load_embeddings("../glove/glove.6B.50d.txt", "../vocabulary_20.txt")
    """
    vocab = read_vocabulary(vocab_path)
    vocab_size = vocab.get_vocab_size()
    words_to_keep = set(vocab.get_index_to_token_vocabulary().values())
    glove_embeddings = {}
    embedding_dim = None

    logger.info("Reading GloVe embeddings from {}".format(glove_path))
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file,
                         total=get_num_lines(glove_path)):
            fields = line.strip().split(" ")
            word = fields[0]
            if word in words_to_keep:
                vector = np.asarray(fields[1:], dtype="float32")
                if embedding_dim is None:
                    embedding_dim = len(vector)
                else:
                    assert embedding_dim == len(vector)
                glove_embeddings[word] = vector

    all_embeddings = np.asarray(list(glove_embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    logger.info("Initializing {}-dimensional pretrained "
                "embeddings for {} tokens".format(
                    embedding_dim, vocab_size))
    embedding_matrix = torch.FloatTensor(
        vocab_size, embedding_dim).normal_(
            embeddings_mean, embeddings_std)
    # Manually zero out the embedding of the padding token (0).
    embedding_matrix[0].fill_(0)
    # This starts from 1 because 0 is the padding token, which
    # we don't want to modify.
    for i in range(1, vocab_size):
        word = vocab.get_token_from_index(i)

        # If we don't have a pre-trained vector for this word,
        # we don't change the row and the word has random initialization.
        if word in glove_embeddings:
            embedding_matrix[i] = torch.FloatTensor(glove_embeddings[word])
    return embedding_matrix, vocab


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

ext = Extractor(0.05, 'ROUGE-2')
# sentences = ['Tokyo is the capital of Japan and the center of Japanese economy.', 'Tokyo is the commerce center of Japan.', 'I like puppies.']
# reference = "The capital of Japan, Tokyo, is the center of Japanese economy."
# print(ext.experimental_similarity(sentences, reference))
# print(ext.experimental_similarity(sentences, reference))
ext.average_word_vectors("the Karishma")