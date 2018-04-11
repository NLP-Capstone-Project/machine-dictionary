
import torch
import torch.nn as nn

"""
TODO:
     i. Flesh out model architecture
        - How to incorporate topics
        - How to incorporate co-variance / co-occurrences of words
        - How to collect semantically significant words for dots apporach
        - Define inference that connects two words
          (word limits? different parameters?)
    ii. Fine-grain details of architecture
        - Define parameters (weights), batch vs. word by word, etc.
   iii. Integrate with data.py (make sure the model knows what to expect)

"""

class DefinitionRNN(nn.module):

    def __init__(self):

        """
        RNN Language Model with latently learned topics for capturing global
        semantics: https://arxiv.org/abs/1611.01702

        When predicting each word, uses likelihood of it being a stopword to
        throttle the inclusion of topics in inference.

        Definitions of Constants
        ------------------------
        C - Vocabulary size including stopwords
        H - RNN hidden size
        K - Number of topic dimensions
        E - Normal Distribution parameters inference network's hidden dimension

        Model Parameters and Dimensions
        -------------------------------
        U :=
        W := RNN f_w weights for calculating h_t (H x H)
        V := RNN weights for inference on y_t(H x C)
        beta := Topic distributions over words (row-major) (K x C)
        theta := Topic vector (K)
        W_1 := Weights for affine calculating mu (E)
        W_2 := Weights for affine calculating sigma (E)

        Python/Torch Parameters:
        -----------
        :param hidden_size: int
            The hidden size of the RNN

        :param topic_dim: int
            The number of topics the model will learn.
        """
