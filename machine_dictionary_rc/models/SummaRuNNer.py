import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

"""
TODO:
    - Batching (30 sentences?)
    - New affines + character-level RNN to encode words.

"""


class SummaRuNNer(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size,
                 position_size=500, position_embedding_size=100,
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
        self.content = nn.Linear(hidden_size * 2, 1, bias=False)
        self.salience = nn.Bilinear(hidden_size * 2, hidden_size * 2, 1,
                                    bias=False)
        self.novelty = nn.Bilinear(hidden_size * 2, hidden_size * 2, 1,
                                   bias=False)
        self.abs_pos = nn.Linear(position_embedding_size, 1, bias=False)
        self.rel_pos = nn.Linear(position_embedding_size, 1, bias=False)

        self.word_rnn = nn.GRU(
                            input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True
                        )

        self.sentence_rnn = nn.GRU(
                                input_size=hidden_size * 2,
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
        sentence_representations = Variable(torch.zeros(len(sentences),
                                            self.hidden_size * 2))

        for i, sentence in enumerate(sentences):
            embedded_sentence = self.embedding(sentence)

            # Pass through Bidirectional word-level RNN with batch size 1.
            word_output, word_hidden = self.word_rnn(embedded_sentence
                                                     .unsqueeze(0))

            # Average pooling across words of the sentence.
            sentence_representations[i] = torch.mean(word_output.squeeze(0), 0)

        # Forward pass on Bidirectional sentence-level RNN with batch size 1.
        sentences_output, sentences_hidden = self.sentence_rnn(
            sentence_representations.unsqueeze(0))

        # Affine on pooled sentence hidden states produces a
        # document representation.
        pooled_sentences = torch.mean(sentences_output.squeeze(), 0)
        encoded_pooled_sentences = self.tanh(self.encode_document(pooled_sentences))
        return sentence_representations, encoded_pooled_sentences

    def forward(self, sentence, index, s_j, doc_len, doc_rep):
        """
        Given a sentence at index 'index' for a given document,
        predicts whether the sentence should be included in the
        current running summary.
        :param sentence: torch.LongTensor
            An encoded sentence to classify.
        :param index: int
            The place in which it occurs in the document.
        :param s_j: torch.FloatTensor
            The current running summary representation.
        :param doc_len: int
            The length of the document in sentences.
        :param doc_rep: torch.FloatTensor
            The average pooling of all sentences in the document.
        :return: The probability of this sentence being included in a summary.
        """
        # Forward pass through the bidirectional GRU.
        # Pass through Bidirectional word-level RNN with batch size 1.
        embedded_sentence = self.embedding(sentence)
        word_output, _ = self.word_rnn(embedded_sentence.unsqueeze(0))
        h_j = torch.mean(word_output.squeeze(), 0)

        abs_index = Variable(torch.LongTensor([index]))
        absolute_pos_embedding = self.abs_pos_embedding(abs_index).squeeze()

        rel_index = int(round(index + 1) * 9.0 / doc_len)  # found on github
        rel_index = Variable(torch.LongTensor([rel_index]))
        relative_pos_embedding = self.rel_pos_embedding(rel_index).squeeze()

        # Classifying the sentence.
        content = self.content(h_j)

        # Salience = h_t^T x W_salience x D
        salience = self.salience(h_j.view(1, -1), doc_rep)

        # Novelty = h_j^T x W_novelty * Tanh(s_j)
        novelty = self.novelty(h_j.view(1, -1), self.tanh(s_j))

        absolute_position_importance = self.abs_pos(absolute_pos_embedding)
        relative_position_importance = self.rel_pos(relative_pos_embedding)

        probabilities = F.sigmoid(content
                                  + salience
                                  - novelty  # Punish for repeating words.
                                  + absolute_position_importance
                                  + relative_position_importance)

        return probabilities, h_j
