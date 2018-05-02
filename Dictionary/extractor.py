
from pythonrouge.pythonrouge import Pythonrouge
import torch


class Extractor(object):
    """
    Extractor class with ROUGE evaluation functionality.

    Example:

    summary = [[" Tokyo is the capital of Japan"], ["Tokyo is the commerce center of Japan"], ["I like puppies"]]
    reference = ["The capital of Japan, Tokyo, is the center of Japanese economy."]

    ext = Extractor(0.05, 'ROUGE-2')

    ext.construct_extraction_from_document(summary, reference) will give us [Tokyo is the capital of Japan]
    """

    def __init__(self, threshold, rouge_type='ROUGE-1'):
        self.threshold = threshold
        self.rouge_type = rouge_type

    def construct_extraction_from_document(self, document_sentences, reference):
        """
        Uses a greedy approach to find the sentences which maximize the ROUGE score
        with respect to the reference definition
        """

        # TODO: Save everything to JSON files.
        extracted = []
        ret_tensor = torch.zeros(len(document_sentences)).long()
        score = 0
        reference = [[[reference]]]
        for i, sentence in enumerate(document_sentences):
            print(i)
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

        return extracted, ret_tensor
