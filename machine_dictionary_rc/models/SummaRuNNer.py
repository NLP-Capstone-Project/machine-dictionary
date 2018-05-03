
from allennlp.nn.util import sort_batch_by_length
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
                 position_size=1000, position_embedding_size=100,
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

    def document_representation(self, document_tensor):
        """
        Compute the sentence representation, D.
        :param document_tensor:
            Stacked tensors of the sentences given throughout the document.
            Assumes document_tensor is wrapped with Variable.
        :return: D: The average pooled representation of the document.
        """

        # 1. Pad variable lengths sentences to prevent the model from learning
        #    from the padding.

        # Collect lengths for sorting and padding.
        # Shape: (batch_size,)
        document_mask = (document_tensor != 0)
        sentence_lengths = Variable(document_mask.sum(dim=1))

        # Shape: (batch_size x max sentence length x embedding size)
        embedded_sentences = self.embedding(Variable(document_tensor))
        sorted_embeddings, sorted_lengths, restore_index, permute_index \
            = sort_batch_by_length(embedded_sentences, sentence_lengths)

        sorted_lengths = list(sorted_lengths.data.long())

        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_embeddings,
                                                             sorted_lengths,
                                                             batch_first=True)

        # 2. Encode the sentences at the word level.
        # Shape: (batch_size x max sentence length x bidirectional hidden)
        #        (batch_size x bidirectional hidden)
        sentences_out, sentences_hidden = self.word_rnn(packed_sentences)

        padded_sentences, padded_sentences_lengths = \
            nn.utils.rnn.pad_packed_sequence(sentences_out, batch_first=True)

        # Restore order for predictions.
        encoded_sentences_restored = padded_sentences[restore_index]

        # 3. Pool along the length dimension.
        sentence_representations = torch.mean(encoded_sentences_restored, 1)

        # 4. Encode the document at the sentence level.
        doc_out, doc_hiddens = self.sentence_rnn(sentence_representations.unsqueeze(0))

        # 4. Average the sentence representations and push through affine.
        pooled_doc_out = torch.mean(doc_out.squeeze(), 0)
        doc_rep = self.encode_document(pooled_doc_out)

        return sentence_representations, doc_rep

    def forward(self, sentence_hidden_states, index, running_summary,
                document_lengths, document_representations):
        """
        Given a sentence at index 'index' for a given document,
        predicts whether the sentence should be included in the
        current running summary.
        :param sentence_hidden_states: torch.FloatTensor
            An encoded sentence to classify.
        :param index: int
            The place in which it occurs in the document.
        :param running_summary: torch.FloatTensor
            The current running summary representation.
        :param doc_len: int
            The length of the document in sentences.
        :param document_representations: torch.FloatTensor
            The average pooling of all sentences in the document.
        :return: The probability of this sentence being included in a summary.
        """
        # Forward pass through the bidirectional GRU.
        # Pass through Bidirectional word-level RNN with batch size 1.
        # By taking the number of sentences rather than the batch size, allows
        # remainders to be included in the calculation.
        abs_index = torch.LongTensor([index] * sentence_hidden_states.size(0))

        # Quantize each document into 10 segments.
        rel_index = ((abs_index.float() / document_lengths.float()) * 10).long()

        # Embed the positions.
        absolute_pos_embedding = self.abs_pos_embedding(Variable(abs_index))
        relative_pos_embedding = self.rel_pos_embedding(Variable(rel_index))

        # Classify the sentence.
        content = self.content(sentence_hidden_states)

        # Salience = h_t^T x W_salience x D
        salience = self.salience(sentence_hidden_states, document_representations)

        # Novelty = h_j^T x W_novelty * Tanh(s_j)
        novelty = self.novelty(sentence_hidden_states, self.tanh(running_summary))

        absolute_position_importance = self.abs_pos(absolute_pos_embedding)
        relative_position_importance = self.rel_pos(relative_pos_embedding)

        probabilities = F.sigmoid(content
                                  + salience
                                  - novelty  # Punish for repeating words.
                                  + absolute_position_importance
                                  + relative_position_importance)

        return probabilities
