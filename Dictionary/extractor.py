import copy

import spacy
import nltk
from nltk.util import ngrams
from pythonrouge.pythonrouge import Pythonrouge
import torch
from tqdm import tqdm


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
        self.nlp = spacy.load('en_core_web_sm')
        self.to_lowercase = to_lowercase
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.obtain_stopwords("stopwords.txt")

    @staticmethod
    def is_subsequence(a, b):
        for i in range(0, len(b)):
            if b[i: i + len(a)] == a:
                return True

        return False

    @staticmethod
    def obtain_stopwords(filename):
        file = open(filename, "r")
        return file.read().splitlines()

    def strip_stopwords(self, sentence):
        sentence_split = sentence.split()
        if self.to_lowercase:
            sentence_split = [word.lower() for word in sentence_split]
            result_words = [word for word in sentence_split if word not in self.stopwords]
        else:
            result_words = [word for word in sentence_split if word not in self.stopwords]

        return ' '.join(result_words)

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
                                skip_threshold=15, cosine_threshold=0.92):
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
                sentence_nlp = self.nlp(self.strip_stopwords(sentences[i].lower()))
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

# ext = Extractor(0.05, 'ROUGE-2')
# sentences = ['Tokyo is the capital of Japan and the center of Japanese economy.', 'Tokyo is the commerce center of Japan.', 'I like puppies.']
# reference = "The capital of Japan, Tokyo, is the center of Japanese economy."
# print(ext.experimental_similarity(sentences, reference))
# print(ext.experimental_similarity(sentences, reference))