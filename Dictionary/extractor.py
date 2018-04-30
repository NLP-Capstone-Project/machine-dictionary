from pythonrouge.pythonrouge import Pythonrouge

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
        extracted = []
        ret_tensor = []
        score = 0
        reference = [[[reference]]]
        for sentence in document_sentences:
            summary = extracted.copy()
            summary.append([sentence])
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=summary, reference=reference,
                                n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                                recall_only=True, stemming=True, stopwords=True,
                                word_level=True, length_limit=True, length=50,
                                use_cf=False, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
            temp_score = rouge.calc_score()
            if (temp_score[self.rouge_type]) > score:
                extracted.append([sentence])
                score = temp_score[self.rouge_type]
                ret_tensor.append(1)
            else:
                ret_tensor.append(0)
        return extracted, torch.LongTensor(ret_tensor)
