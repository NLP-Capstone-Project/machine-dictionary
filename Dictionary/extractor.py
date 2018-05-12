
import en_core_web_sm
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

    def __init__(self, threshold, rouge_type='ROUGE-1', n_gram=2):
        self.threshold = threshold
        self.rouge_type = rouge_type
        self.n_gram = n_gram
        self.nlp = en_core_web_sm.load()

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

    def cosine_similarity(self, sentences, reference,
                                 threshold=0.5, delta=0.1):
        """
        Collect sentences greedily using cosine similarity as the heuristic.
        :param sentences: List of list of words representing the document.
        :param reference: The reference to the UMLS term to define.
        :param threshold: Minimum cosine similarity score in order to be included.
        :return: A tensor where 1 means a sentence should be included.
        """
        reference = self.nlp(reference)
        ret_tensor = torch.zeros(len(sentences)).long()
        score = 0.0
        extracted = []
        for i in tqdm(range(len(sentences))):
            extracted.append(i)  # consider the sentence
            summary = ""  # generate a running summary including the current sentence
            for sentence in extracted:
                summary += sentences[sentence] + " "
            cosine_similarity = reference.similarity(self.nlp(summary))
            if cosine_similarity >= threshold and cosine_similarity - score > delta:
                ret_tensor[i] = 1
                score = cosine_similarity
            else:
                extracted = extracted[:len(extracted) - 1]
        return ret_tensor

    def skipgram_similarity(selfself, skipgram_map, reference):
        return None

    def construct_skipgram_map(self, sentences):
        skipgram_map = {}
        for i, sentence in enumerate(sentences):
            skipgram_map[i] = set()
            words = sentence.split(' ')
            sentence_len = len(words)
            for j in range(sentence_len):
                for k in range(j + 1, sentence_len):
                    skipgram_map[i].add((words[j], words[k]))
        return skipgram_map


ext = Extractor(0.05, 'ROUGE-2')
#
sentences = ['Tokyo is the capital of Japan and the center of Japanese economy.', 'Tokyo is the commerce center of Japan.', 'I like puppies.']
# reference = "The capital of Japan, Tokyo, is the center of Japanese economy."
#
# print(ext.cosine_similarity(sentences, reference))
print(ext.construct_skipgram_map(sentences))