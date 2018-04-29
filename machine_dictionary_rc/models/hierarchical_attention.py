from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import torch.nn as nn


class HAN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size,
                 layers=1, dropout=0.5):

        """
        Hierarchical Attention Network for encoding documents
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(GRU, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.layers = layers

        # Learned word embeddings (vocab_size x embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # GRU, accepts vectors of length 'embedding_size'.
        self.rnn = nn.GRU(embedding_size, hidden_size, layers,
                          dropout=dropout,
                          batch_first=True, bidirectional=True)

        self.word_gru = nn.GRU(embedding_size, hidden_size, layers, droput=dropout, batch_first=True)
        self.sentence_gru = nn.GRU(embedding_size * 2, hidden_size, layers, dropout=dropout, batch_first=True)

        self.word_context = nn.Linear(hidden_size * 2, 1)
        self.sentence_context = nn.Linear(hidden_size * 2, 1)

    def init_hidden(self):
        """
        Produce a new, initialized hidden state variable where all values
        are zero.
        :return: A torch Tensor.
        """

        weight = next(self.parameters()).data
        return Variable(weight.new(self.layers, self.batch_size, self.hidden_size).zero_())

    def forward(self, input, hidden):
        # Embed the input
        # Shape: (batch, length (single word), embedding_size)
        embedded_input = self.embedding(input).view(self.batch_size, 1, -1)

        # Forward pass.
        # Shape (output): (1, hidden_size)
        # Shape (hidden): (layers, batch, hidden_size)
        output, hidden = self.rnn(embedded_input, hidden)

        # Decode the final hidden state
        # Shape: (1, 1)
        decoded = self.decoder(output)

        return decoded, hidden

    def forward(self, forward, backward):
        bi_word_outputs = []
        for i in range(len(forward)):
            bi_word_outputs.append(self.word_encoder(forward[i], backward[i]))
        sentence_vectors = []
        for i in range(len(bi_word_outputs)):
            sentence_vectors.append(self.word_attention(bi_word_outputs))
        bi_sentence_outputs = self.sentence_encoder(sentence_vectors)
        document_vector = self.sentence_attention(bi_sentence_outputs)
        prediction = self.decisionboundary(document_vector)
        return prediction


    def word_encoder(self, forward, backward):
        f_output = []
        hidden = self.init_hidden()
        for i in range(len(forward)):
            embedded = self.embedding(forward[i]).view(1, 1, -1)
            output, hidden = self.word_gru(embedded, hidden)
            f_output.append(hidden)
        b_output = []
        hidden = self.init_hidden()
        for i in range(len(backward)):
            embedded = self.embedding(backward[i]).view(1, 1, -1)
            output, hidden = self.word_gru(embedded, hidden)
            b_output.append(hidden)
        final_output = []
        for i in range(len(forward)):
            final_output.append(torch.cat((f_output[i], b_output[len(backward) - 1 - i]), 2))
        return final_output

    def word_attention(self, sentence_representation):
        attention_coefficients = []
        for i in range(len(sentence_representation)):
            score = torch.exp(self.word_context(sentence_representation[i][0]))
            attention_coefficients.append(score)

        final_vector = Variable(torch.zeros(1, self.hidden_size * 2))
        for i in range(len(sentence_representation)):
            added = (attention_coefficients[i] * sentence_representation[i][0])
            final_vector = final_vector + added

        return final_vector

    def sentence_encoder(self, sentence_vectors):
        f_output = []
        hidden = self.init_hidden()
        for i in range(len(sentence_vectors)):
            output, hidden = self.sentence_gru(sentence_vectors[i], hidden)
            f_output.append(hidden)
        b_output = []
        hidden = self.init_hidden()
        for i in range(len(sentence_vectors) - 1, -1, -1):
            output, hidden = self.sentence_gru(sentence_vectors[i], hidden)
            b_output.append(hidden)
        final_output = []
        for i in range(len(sentence_vectors)):
            final_output.append(torch.cat((f_output[i], b_output[len(b_output) - 1 - i]), 2))
        return final_output

    def sentence_attention(self, sentence_vectors):
        attention_coefficients = []
        for i in range(len(sentence_vectors)):
            score = torch.exp(self.sentence_context(sentence_vectors[i][0]))
            attention_coefficients.append(score)
        final_vector = Variable(torch.zeros(1, self.hidden_size * 2))
        for i in range(len(sentence_vectors)):
            added = (attention_coefficients[i] * sentence_vectors[i][0])
            final_vector = final_vector + added

        return final_vector
