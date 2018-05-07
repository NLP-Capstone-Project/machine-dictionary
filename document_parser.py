import argparse
from collections import Counter
import dill
import logging
import os
import shutil
from tqdm import tqdm
import sys

import pdb

import torch
from torch.autograd import Variable
from torch.nn.functional import cross_entropy, log_softmax, nll_loss

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Dictionary import Corpus, UMLSCorpus,\
    word_vector_from_seq, extract_tokens_from_json, UMLS, Extractor
from machine_dictionary_rc.models.baseline_rnn import RNN
from machine_dictionary_rc.models.baseline_gru import GRU
from machine_dictionary_rc.models.baseline_lstm import LSTM
from machine_dictionary_rc.models.SummaRuNNer import SummaRuNNer
from machine_dictionary_rc.models.SummaRuNNerChar import SummaRuNNerChar

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
    "vanilla": RNN,
    "gru": GRU,
    "lstm": LSTM,
    "tagger": SummaRuNNerChar
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--train-path", type=str,
                        default=os.path.join(
                            project_root, "data", "train"),
                        help="Path to the Semantic Scholar training data.")
    parser.add_argument("--parsed-path", type=str,
                        default=os.path.join(
                            project_root, "data", "test"),
                        help="Path in which to save the parsed data.")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(
                            project_root, "data"),
                        help="Path to the directory containing the data.")
    parser.add_argument("--built-corpus-path", type=str,
                        default=os.path.join(
                            project_root, "corpus.pkl"),
                        help="Path to a pre-constructed corpus.")
    parser.add_argument("--built-umls-path", type=str,
                        default=os.path.join(
                            project_root, "umls.pkl"),
                        help="Path to a pre-constructed umls dataset.")
    parser.add_argument("--passage-testing-length", type=int, default=200,
                        help="Number of words to encode for feature extraction.")
    parser.add_argument("--definitions-path", type=str,
                        help="Path to the UMLS MRDEF.RRF file.")
    parser.add_argument("--synonyms-path", type=str,
                        help="Path to the UMLS MRCONSO.RRF file.")
    parser.add_argument("--rouge-threshold", type=float,
                        default=0.1,
                        help="ROUGE threshold")
    parser.add_argument("--rouge-type", type=str,
                        default='ROUGE-2',
                        help="ROUGE type")
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--model-type", type=str, default="vanilla",
                        choices=["vanilla", "lstm", "gru", "tagger"],
                        help="Model type to train.")
    parser.add_argument("--min-token-count", type=int, default=5,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--bptt-limit", type=int, default=50,
                        help="Extent in which the model is allowed to"
                             "backpropagate.")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and TopicRNN models.")
    parser.add_argument("--embedding-size", type=int, default=50,
                        help="Embedding size to use in RNN and TopicRNN models.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.005,
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

    if args.model_type not in MODEL_TYPES:
        raise ValueError("Please select a supported model.")

    if not args.train_path:
        raise ValueError("Training data directory required")

    # Set the groundwork for constructing a UMLS corpus.
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    # Explicit dir just for parsed documents.
    parsed_dir = os.path.join(args.data_dir, "parsed")
    if not os.path.exists(parsed_dir):
        os.mkdir(args.parsed_path)

    # Explicit dir for just BIO-tagged sentences.
    bio_dir = os.path.join(args.data_dir, "BIO")
    if not os.path.exists(bio_dir):
        if not os.path.exists(bio_dir):
            os.mkdir(bio_dir)

    print("Restricting vocabulary based on min token count",
          args.min_token_count)

    print("Collecting Semantic Scholar JSONs:")

    dictionary = process_corpus(args.train_path, args.parsed_path, args.min_token_count)
    vocab_size = len(dictionary)
    print("Vocabulary Size:", vocab_size)

    # Extract references from UMLS
    try:
        umls = UMLS(args.definitions_path, args.synonyms_path)
        umls.generate_all_definitions()
        extractor = Extractor(args.rouge_threshold, args.rouge_type)
        umls_dataset = UMLSCorpus(dictionary, extractor, umls, bio_dir,
                                  batch_size=args.batch_size)
        umls_dataset.generate_all_data()
        print()  # Printing in-place progress flushes standard out.
    except KeyboardInterrupt:
        print("Stopping BIO tagging early.")
        sys.exit()


def process_corpus(train_path, parsed_path, min_token_count):
    """
    Parses and saves Semantic Scholar JSONs found in 'train_path'.
    :param train_path: file path
        The path to the JSON documents meant for training / validation.
    :param parsed_path: file path
        The directory in which each processed document is saved in.
    :param min_token_count:
        The minimum number of times a word has to occur to be included.
    """
    all_training_examples = os.listdir(train_path)

    tokens = []
    for file in tqdm(all_training_examples):
        file_path = os.path.join(train_path, file)
        tokens += extract_tokens_from_json(file_path)

    # Map words to the number of times they occur in the corpus.
    word_frequencies = dict(Counter(tokens))

    # Sieve the dictionary by excluding all words that appear fewer
    # than min_token_count times.
    vocabulary = set([w for w, f in word_frequencies.items()
                      if f >= min_token_count])

    # Construct the corpus with the given vocabulary.
    corpus = Corpus(vocabulary)

    print("Saving sentence-parsed Semantic Scholar JSON files (training):")
    for file in tqdm(all_training_examples):
        # Corpus expects a full file path.
        corpus.save_processed_document(os.path.join(train_path, file),
                                       os.path.join(parsed_path, file))

    return corpus


def get_document_tensor(sentences):
    max_length = len(max(sentences, key=lambda x: len(x)))
    sentences_tensor = torch.zeros(len(sentences), max_length).long()
    for j, sentence in enumerate(sentences):
        sentences_tensor[j, :len(sentence)] = sentence

    return sentences_tensor


def print_progress_in_place(*args):
    print("\r", *args, end="")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
