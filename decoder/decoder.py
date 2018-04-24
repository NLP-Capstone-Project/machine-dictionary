
import operator

import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax

class Decoder(object):
    def __init__(self, model, fsm, words_to_sentences, dictionary, beam=50):
        self.dictionary = dictionary
        self.model = model
        self.fsm = fsm
        self.words_to_sentences = words_to_sentences
        self.beam = beam

    def decode(self, seed, sentence_limit=50):
        """
        Given a seed in which to generate from, computes the
        most likely sentence under the model given ontological constraints.
        :param seed:
            A semantically significant word relevant to the term to be defined.
        :return: A sentence containing the seed.
        """

        # Should we have another scoring mechanism on top of this?
        # TODO: Visit Yejin if possible.

        # Stop early if the predicted token is an EOS token.

    def top_k_transitions(self, word, hidden):
        hidden_copy = self.clone_hidden(hidden)
        word_vector = torch.LongTensor(1)
        word_encoded = self.dictionary.word_to_index[word]
        word_vector[0] = word_encoded

        output, hidden_new = self.model(Variable(word_vector), hidden_copy)

        # Calculate probability of the next word given the model
        # in log space.
        # Reshape to (vocab size x 1) and perform log softmax over
        # the first dimension.
        prediction_probabilities = log_softmax(output.view(-1, 1), 0)

        # Indices represent the next words to traverse
        values, indices = prediction_probabilities.topk(self.beam)
        indices_as_words = [(self.dictionary.index_to_word[i], values[i])
                            for i in indices]

        # But we can only select those in our restricted vocabulary.
        indices_as_words = [(w, v) for (w, v) in indices_as_words
                            if w in self.fsm.allowable_states]

        # Sort according to beam size
        indices_as_words = sorted(indices_as_words, key=lambda x: x[1])

        return indices_as_words[:self.beam]







    def clone_hidden(self, hidden):
        return Variable(hidden.data.clone())
