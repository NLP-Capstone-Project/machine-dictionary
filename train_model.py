import argparse
import logging
import os
import shutil
from tqdm import tqdm
import sys

import torch
from torch.autograd import Variable
from torch import optim
from torch.nn.functional import cross_entropy

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Dictionary import Corpus, word_vector_from_seq
from machine_dictionary_rc.models.rnn import RNN

logger = logging.getLogger(__name__)

"""
TODO:
    Training
    Evaluation
    Logging
    Define format of Data
        - Should be separated by paragraph for time
    Define Evaluation Metrics
        - Sentiment Analysis?
        - Perplexity?
"""

MODEL_TYPES = {
    "vanilla": RNN
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--train-path", type=str,
                        default=os.path.join(
                            project_root, "data", "conflicts", "content"),
                        help="Path to the conflict training data.")
    parser.add_argument("--dev-path", type=str,
                        default=os.path.join(
                            project_root, "conflicts", "validation"),
                        help="Path to the conflicts dev data.")
    parser.add_argument("--test-path", type=str,
                        default=os.path.join(
                            project_root, "conflicts", "test"),
                        help="Path to the conflicts test data.")
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--model-type", type=str, default="topic-rnn",
                        choices=["topic", "vanilla"],
                        help="Model type to train.")
    parser.add_argument("--min-token-count", type=int, default=10,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and TopicRNN models.")
    parser.add_argument("--embedding-size", type=int, default=50,
                        help="Embedding size to use in RNN and TopicRNN models.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="The learning rate to use.")
    parser.add_argument("--log-period", type=int, default=50,
                        help=("Update training metrics every "
                              "log-period weight updates."))
    parser.add_argument("--validation-period", type=int, default=500,
                        help=("Calculate metrics on validation set every "
                              "validation-period weight updates."))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to use")
    parser.add_argument("--cuda", action="store_true",
                        help="Train or evaluate with GPU.")
    args = parser.parse_args()

    # Set random seed for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("\033[35mGPU available but not running with "
                           "CUDA (use --cuda to turn on.)\033[0m")
        else:
            torch.cuda.manual_seed(args.seed)

    # Construct vocabulary
    corpus = Corpus()

    if not args.conflict_train_path:
        raise ValueError("Training data directory required")

    # TODO: Get Corpus working on all docs.
    # TODO: Make a vocabulary that restricts the vocab size.
    train_path = args.conflict_train_path
    corpus.add_example(train_path)
    vocab_size = len(corpus.dictionary)

    # Create model of the correct type.
    if args.model_type == "topic":
        logger.info("Building TopicRNN model")
        model = TopicRNN(vocab_size, args.embedding_size, args.hidden_size, args.dropout)
    else:
        logger.info("Building Elman RNN model")
        model = RNN(vocab_size, args.embedding_size, args.hidden_size,
                    layers=2, dropout=args.dropout)

    logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    train_epoch(model, corpus, 0, optimizer)


def train_epoch(model, corpus, batch_size, optimizer):
    """
    Train the model for one epoch.
    """

    # Set model to training mode (activates dropout and other things).
    model.train()
    for document in corpus.documents:
        # Incorporation of time requires feeding in by one word at
        # a time.
        #
        # Iterate through the words of the document, calculating loss between
        # the current word and the next, from first to penultimate.

        loss = 0
        hidden = model.init_hidden()
        for i in tqdm(range(document.size(0) - 1)):
            current_word = Variable(word_vector_from_seq(document, i))
            next_word = Variable(word_vector_from_seq(document, i + 1))
            output, hidden = model(current_word, hidden)

            # Calculate loss between the next word and what was anticipated.
            loss += cross_entropy(output, next_word)

        # Perform backpropagation and update parameters.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()



if __name__ == "__main__":
    main()
