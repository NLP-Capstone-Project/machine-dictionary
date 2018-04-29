import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SummaRuNNer(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size,
                 position_size=128, position_embedding_size=100,
                 layers=1, dropout=0.5):

        """
        SummaRuNNer: A neural-based sentence classifier for Extractive Summarization.

        Parameters:
        -----------
        :param vocab_size: int
            The embedding size for embedding input words (space in which
            words are projected).

        :param embedding_size: int
            The embedding size for embedding input words (space in which
            words are projected).

        :param hidden_size: int
            The hidden size of the bi-directional GRU.

        :param position_size: int
            The length of the longest document in sentences.

        :param position_embedding_size: int
            The embedding size for absolute and relative position embeddings.
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(SummaRuNNer, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.layers = layers
        self.position_embedding_size = position_embedding_size
        self.position_size = position_size

        # Activations
        self.tanh = nn.Tanh()

        # Learned word embeddings (vocab_size x embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Positional embeddings
        self.abs_pos_embedding = nn.Embedding(position_size,
                                              position_embedding_size)
        self.rel_pos_embedding = nn.Embedding(position_size,
                                              position_embedding_size)

        # SummaRuNNer coherence affine transformations.
        self.content = nn.Linear(hidden_size * 2, 1,
                                 bias=False)
        self.salience = nn.Bilinear(hidden_size * 2, hidden_size * 2, 1,
                                    bias=False)
        self.novelty = nn.Bilinear(hidden_size * 2, hidden_size * 2, 1,
                                   bias=False)
        self.abs_pos = nn.Linear(position_embedding_size, 1, bias=False)
        self.rel_pos = nn.Linear(position_embedding_size, 1, bias=False)

        self.word_forward = nn.GRU(
                                input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=layers,
                                dropout=dropout,
                                batch_first=True
                            )

        self.word_reverse = nn.GRU(
                                input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=layers,
                                dropout=dropout,
                                batch_first=True
                            )

        self.sentence_lvl_rnn = nn.GRU(
                                    input_size=embedding_size,
                                    hidden_size=hidden_size,
                                    num_layers=layers,
                                    dropout=dropout,
                                    batch_first=True, bidirectional=True
                                 )

        # Encoders and Decoders
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.encode_document = nn.Linear(hidden_size * 2, hidden_size * 2)

    def init_hidden(self):
        """
        Produce a new, initialized hidden state variable where all values
        are zero.
        :return: A torch Tensor.
        """

        weight = next(self.parameters()).data
        return Variable(weight.new(self.layers, self.batch_size,
                                   self.hidden_size).zero_())

    def document_representation(self, sentences):
        """
        Compute the sentence representation, D.
        :param sentences:
            The list of sentence tensors given throughout the document.
        :return: D: The average pooled representation of the document.
        """
        average_hidden = torch.zeros(self.hidden_size * 2)
        for sentence in sentences:
            average_hidden += self.bidirectional_hidden_state(sentence)

        average_hidden /= len(sentences)

        return self.tanh(self.encode_document(Variable(average_hidden)))

    def forward(self, sentence, index, s_j, doc_len, doc_rep):

        # Forward pass through the bidirectional GRU.
        # Shape: (batch_size, hidden_size * 2)
        h_j = self.bidirectional_hidden_state(sentence)

        abs_index = Variable(torch.LongTensor([index]))
        absolute_pos_embedding = self.abs_pos_embedding(abs_index).squeeze()

        rel_index = int(round(index + 1) * 9.0 / doc_len)  # found on github
        rel_index = Variable(torch.LongTensor([rel_index]))
        relative_pos_embedding = self.rel_pos_embedding(rel_index).squeeze()

        # Classifying the sentence.
        h_j = Variable(h_j)
        content = self.content(h_j)

        # Salience = h_t^T x W_salience x D
        salience = self.salience(h_j.view(1, -1), doc_rep)

        # Novelty = h_j^T x W_novelty * Tanh(s_j)
        novelty = self.novelty(h_j.view(1, -1), self.tanh(s_j))

        absolute_position_importance = self.abs_pos(absolute_pos_embedding)
        relative_position_importance = self.rel_pos(relative_pos_embedding)

        probabilities = F.sigmoid(content +
                                  salience +
                                  novelty +
                                  absolute_position_importance +
                                  relative_position_importance)
        return probabilities, h_j

    def bidirectional_hidden_state(self, sentence):
        embed_forward = self.embedding(Variable(sentence))

        # Reverse direction.
        reverse_indices = list(range(len(sentence)))[::-1]
        embed_reverse = self.embedding(Variable(sentence[reverse_indices]))

        # Encode the sentence using a bidirectional GRU.
        _, h_forward = self.word_forward(embed_forward
                                         .view(self.batch_size,
                                               embed_forward.size(0), -1))

        _, h_reverse = self.word_forward(embed_reverse
                                         .view(self.batch_size,
                                               embed_reverse.size(0), -1))

        # Flatten to 1D tensors.
        h_forward = h_forward.data[0].squeeze()
        h_reverse = h_reverse.data[0].squeeze()

        return torch.cat((h_forward, h_reverse))
