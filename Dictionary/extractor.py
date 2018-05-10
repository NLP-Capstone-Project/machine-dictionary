
from pythonrouge.pythonrouge import Pythonrouge
import torch
from tqdm import tqdm
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

class Extractor(object):
    """
    Extractor class with ROUGE evaluation functionality.

    Example:

    summary = [[" Tokyo is the capital of Japan"], ["Tokyo is the commerce center of Japan"], ["I like puppies"]]
    reference = ["The capital of Japan, Tokyo, is the center of Japanese economy."]

    ext = Extractor(0.05, 'ROUGE-2')

    ext.construct_extraction_from_document(summary, reference) will give us [Tokyo is the capital of Japan]



    Example execution of how a new metric might be faster. Uncomment to run

    ext = Extractor(0.05, 'ROUGE-2')

    summary = [" Tokyo is the capital of Japan", "Tokyo is the commerce center of Japan", "I like puppies"]
    reference = "The capital of Japan, Tokyo, is the center of Japanese economy."

    summaryROUGE = [[" Tokyo is the capital of Japan"], ["Tokyo is the commerce center of Japan"], ["I like puppies"]]
    referenceROUGE = ["The capital of Japan, Tokyo, is the center of Japanese economy."]

    ext.extraction_unigram(summary, reference)
    ext.construct_extraction_from_document(summaryROUGE, referenceROUGE)

    """

    def __init__(self, threshold, rouge_type='ROUGE-1', n_gram=2):
        self.threshold = threshold
        self.rouge_type = rouge_type
        self.n_gram = n_gram

    def construct_extraction_from_document(self, document_sentences, reference):
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

    def extraction_ngram(self, document_sentences, reference):
        """
        Uses a greedy approach to find the sentences which maximize the ngram similarity
        with respect to the reference definition.
        """
        sentence_to_ngram = self.construct_sentence_ngram_map(document_sentences)

        reference_ngrams = set()
        token = nltk.word_tokenize(reference)
        n_grams = ngrams(token, self.n_gram)
        for gram in n_grams:
            reference_ngrams.add(gram)

        extracted = []
        ret_tensor = torch.zeros(len(document_sentences)).long()

        for i, sentence in tqdm(enumerate(document_sentences)):
            extracted.append(i)
            intersection = reference_ngrams
            for sentence in extracted:
                intersection &= sentence_to_ngram[sentence]
            if len(intersection) > 0:
                ret_tensor[i] = 1
            else:
                extracted = extracted[:len(extracted) - 1]
        return ret_tensor

    def construct_sentence_ngram_map(self, document_sentences):
        """
        Constructs a map from sentences to sets of ngrams, useful for precomputation
        """
        sentence_to_ngram = {}
        for i, sentence in enumerate(document_sentences):
            sentence_to_ngram[i] = set()
            token = nltk.word_tokenize(sentence)
            n_grams = ngrams(token, self.n_gram)
            for gram in n_grams:
                sentence_to_ngram[i].add(gram)
        return sentence_to_ngram

#
# ext = Extractor(0.05, 'ROUGE-2', 2)
#
# summary = [" Tokyo is the capital of Japan", "Tokyo is the commerce center of Japan", "I like puppies"]
# reference = "The capital of Japan, Tokyo, is the center of Japanese economy."
#
# summaryROUGE = [[" Tokyo is the capital of Japan"], ["Tokyo is the commerce center of Japan"], ["I like puppies"]]
# referenceROUGE = ["The capital of Japan, Tokyo, is the center of Japanese economy."]
#
# print(ext.extraction_ngram(summary, reference))
# print(ext.construct_extraction_from_document(summaryROUGE, referenceROUGE))
