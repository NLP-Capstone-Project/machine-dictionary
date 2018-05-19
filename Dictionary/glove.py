import logging
import mmap

from allennlp.data import Vocabulary
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_vocabulary(vocab_path):
    train_vocab = Vocabulary()
    vocab_file = open(vocab_path, "r")
    vocab = set([word.strip() for word in vocab_file.readlines()])
    for word in vocab:
        train_vocab.add_token_to_namespace(word)
    return train_vocab


def load_embeddings(glove_path, vocab_path):
    """
    Create an embedding matrix for a Vocabulary.
    Usage: load_embeddings("../glove/glove.6B.50d.txt", "../vocabulary_20.txt")
    """
    vocab = read_vocabulary(vocab_path)
    vocab_size = vocab.get_vocab_size()
    words_to_keep = set(vocab.get_index_to_token_vocabulary().values())
    glove_embeddings = {}
    embedding_dim = None

    logger.info("Reading GloVe embeddings from {}".format(glove_path))
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
    logger.info("Initializing {}-dimensional pretrained "
                "embeddings for {} tokens".format(
                    embedding_dim, vocab_size))
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


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
