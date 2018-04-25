
import torch


def word_vector_from_seq(sequence_tensor, i):
    """
    Collect the vector representation of the ith word in a sequence tensor.
    :param sequence_tensor: The document in which to collect from.
    :param i: The ith index of the document.
    :return: A `torch.LongTensor()` containing the document's ith word's
        encoding relative to this corpus.
    """

    word = torch.LongTensor(1)
    word[0] = sequence_tensor[i]
    return word

def load_embeddings(glove_path, vocab):
    """
    Create an embedding matrix for a Vocabulary.
    """
    vocab_size = vocab.get_vocab_size()
    words_to_keep = set(vocab.get_index_to_token_vocabulary().values())
    glove_embeddings = {}
    embedding_dim = None

    with open(glove_path) as glove_file:
        for line in tqdm(glove_file,
                         total=get_num_lines(glove_path)):
            fields = line.strip().split(" ")
            word = fields[0]
            if word in words_to_keep:
                vector = np.asarray(fields[1:], dtype="float32")
                if embedding_dim is None:
                    embedding_dim = len(vector)
                else:
                    assert embedding_dim == len(vector)
                glove_embeddings[word] = vector

    all_embeddings = np.asarray(list(glove_embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    embedding_matrix = torch.FloatTensor(
        vocab_size, embedding_dim).normal_(
            embeddings_mean, embeddings_std)
    # Manually zero out the embedding of the padding token (0).
    embedding_matrix[0].fill_(0)
    # This starts from 1 because 0 is the padding token, which
    # we don't want to modify.
    for i in range(1, vocab_size):
        word = vocab.get_token_from_index(i)

        # If we don't have a pre-trained vector for this word,
        # we don't change the row and the word has random initialization.
        if word in glove_embeddings:
            embedding_matrix[i] = torch.FloatTensor(glove_embeddings[word])
    return embedding_matrix
