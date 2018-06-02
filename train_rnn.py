import argparse
from collections import Counter
import dill
import logging
import os
import shutil
from tqdm import tqdm
import sys

import torch
from torch.nn.functional import cross_entropy

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Dictionary import Dictionary, UMLSCorpus,\
    extract_tokens_from_json, UMLS, Extractor
from machine_dictionary_rc.models.baseline_rnn import RNN
from machine_dictionary_rc.models.baseline_gru import GRU
from machine_dictionary_rc.models.baseline_lstm import LSTM
from machine_dictionary_rc.models.SummaRuNNerChar import SummaRuNNerChar

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
    parser.add_argument("--dev-path", type=str,
                        default=os.path.join(
                            project_root, "data", "validation"),
                        help="Path to the Semantic Scholar dev data.")
    parser.add_argument("--test-path", type=str,
                        default=os.path.join(
                            project_root, "data", "test"),
                        help="Path to the Semantic Scholar test data.")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(
                            project_root, "data"),
                        help="Path to the directory containing the semantic scholar data.")
    parser.add_argument("--bio-dir", type=str,
                        default=os.path.join(
                            project_root, "data", "BIO"),
                        help="Path to the directory containing the annotated data.")
    parser.add_argument("--parsed-dir", type=str,
                        default=os.path.join(
                            project_root, "data", "parsed"),
                        help="Path to the directory containing the parsed data.")
    parser.add_argument("--built-validation-path", type=str,
                        default=os.path.join(
                            project_root, "validation.pkl"),
                        help="Path to a pre-constructed validation dataset.")
    parser.add_argument("--built-dictionary-path", type=str,
                        default=os.path.join(
                            project_root, "dictionary.pkl"),
                        help="Path to a pre-constructed dictionary.")
    parser.add_argument("--vocabulary-path", type=str,
                        default=os.path.join(
                            project_root, "vocabulary.txt"),
                        help="Path to a list of words representing the vocabulary.")
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
    parser.add_argument("--hidden-size", type=int, default=128,
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

    if not os.path.exists(args.dev_path):
                raise ValueError("No directory for validation data given.")

    if not os.path.exists(args.vocabulary_path):
        print("Restricting vocabulary based on min token count",
              args.min_token_count)

        print("Constructing Dictionary:")
        dictionary = init_dictionary(args.train_path, args.min_token_count)
    else:
        with open(args.vocabulary_path, 'r') as f:
            vocab = set([word.strip() for word in f.readlines()])

        dictionary = Dictionary(vocab)

    vocab_size = len(dictionary)
    print("Vocabulary Size:", vocab_size)

    # Create model of the correct type.
    print("Building {} RNN model ------------------".format(args.model_type))
    logger.info("Building {} RNN model".format(args.model_type))
    model = MODEL_TYPES[args.model_type](vocab_size, args.embedding_size,
                                         args.hidden_size, args.batch_size, device,
                                         layers=1, dropout=args.dropout).to(device)

    logger.info(model)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.model_type == "tagger":
        try:
            # Load UMLs / Semantic Scholar Triplets.
            #
            # Contains functionality for constructing the triplets if needed.
            umls = UMLS(args.definitions_path, args.synonyms_path)
            umls.generate_all_definitions()
            extractor = Extractor(args.rouge_threshold, args.rouge_type)
            umls_dataset = UMLSCorpus(dictionary, extractor, umls)

            # Make this a list of directories!
            paper_directories = [os.path.join(args.train_path, paper_dir)
                                 for paper_dir in os.listdir(args.train_path)]
            validation_directories = [os.path.join(args.dev_path, paper_dir)
                                      for paper_dir in os.listdir(args.dev_path)]
            validation_dataset = UMLSCorpus(dictionary, extractor, umls)

            print("Collecting validation data")
            if os.path.exists(args.built_validation_path):
                with open(args.built_validation_path, "rb") as f:
                    validation_dataset = dill.load(f)
            else:
                for valid_dir in tqdm(validation_directories):
                    validation_dataset.collect_all_data(valid_dir)

                with open(args.built_validation_path, "wb") as f:
                    dill.dump(validation_dataset, f)

            print("Training on", len(paper_directories), "documents...")

            # Train the sequence tagger.
            for epoch in range(args.num_epochs):
                for paper_dir in paper_directories:
                    umls_dataset.collect_all_data(paper_dir)
                    train_rnn_epoch(model, umls_dataset, validation_dataset,
                                    args.batch_size, optimizer,
                                    args.save_dir, epoch, log_period=args.log_period)

            print()  # Printing in-place progress flushes standard out.
        except KeyboardInterrupt:
            print("\nStopped training early.")
            pass


def train_rnn_epoch(model, umls_dataset, validation_dataset,
                    batch_size, optimizer,
                    save_dir, epoch, bptt_limit=35, log_period=5):
    """
    Train the tagger model for one epoch.
    """

    model.train()
    print("Training in progress:\n")
    train_loader = umls_dataset.training_loader(batch_size)
    log = open(os.path.join(save_dir, "log.txt"), "w")
    for iteration, batch in enumerate(train_loader):
        # Each example in batch consists of
        # entity: The query term.
        # document: In the document JSON format from 'dictionary'.
        # targets: A list of ints the same length as the number of sentences
        # in the document.
        document_lengths = torch.Tensor([len(ex["sentences"])
                                         for ex in batch]).long()
        max_doc_length = torch.max(document_lengths)

        document_encodings = []
        terms = []
        for i, example in tqdm(list(enumerate(batch))):
            # Compute document representations for each document in 'batch'.
            # Encode the sentences to vector space before inference.
            terms.append(example["entity"])
            sentences = example["sentences"]
            encoded_sentences = validation_dataset.vectorize_sentences(sentences)
            document_tensor = get_document_tensor(encoded_sentences)
            document_tensor = document_tensor.view(-1, )

            # Each row will be a separate document.
            document_encodings.append(document_tensor)

        max_word_length = max(document_encodings, key=lambda x: x.size(0)).size(0)
        batch_document_text = torch.zeros(len(batch), max_word_length).long().to(device)
        for i, encoded_document in enumerate(document_encodings):
            batch_document_text[i][: len(encoded_document)] = encoded_document

        print("Char-level RNN training:")
        batch_char_loss = 0
        for i, term in tqdm(enumerate(terms)):
            if len(term) > 3:  # Three is the smallest for multiple predictions per term.
                input = term[:-1]
                target = model.character_target(term[1:]).long()
                predictions = model.char_level_forward(input)

                # Some words may be two letters long.
                if predictions.dim() < 2:
                    predictions = predictions.unsqueeze(0)

                optimizer.zero_grad()
                loss = cross_entropy(predictions, target)
                loss.backward()
                optimizer.step()

                batch_char_loss += (loss.data.item() / len(term))

        print("Word-level RNN training:")
        batch_word_loss = 0
        for i in tqdm(range(max_doc_length - bptt_limit - 1)):
            import pdb
            pdb.set_trace()
            input = batch_document_text[:, i: i + bptt_limit]
            target = batch_document_text[:, i + 1: i + 1 + bptt_limit]

            # Each batch contributes 'batch_size' predictions per sentence.
            # Shape: (batch_size,)
            predictions = model.word_level_forward(input)

            optimizer.zero_grad()
            loss = cross_entropy(predictions, target)
            loss.backward()
            optimizer.step()

            batch_word_loss += loss.data

        batch_char_loss /= len(batch)
        batch_word_loss /= len(batch)
        print_progress_in_place("Batch #", iteration,
                                "\nAveraged Word Loss for the batch:",
                                batch_word_loss.data.item(),
                                "\nAveraged Char Loss for the batch:",
                                batch_char_loss.data.item(),
                                )
        print()

        if iteration % log_period == 0:
            evaluation = evaluate(model, validation_dataset)
            model.train()
            validation_loss = evaluation["loss"]
            accuracy = evaluation["accuracy"]
            print("Epoch: {} Batch: {}".format(epoch, iteration), file=log)
            print("\tLoss: {}".format(validation_loss), file=log)
            print("\tAccuracy: {}".format(accuracy), file=log)
            save_name = "model_epoch{}_batch{}.ph".format(epoch, iteration)
            save_model(model, save_dir, save_name)


def evaluate(model, validation_dataset, bptt_limit=35):
    """ Evaluate the language model using loss. """
    batch_size = 150
    model.eval()
    print("------------Computing validation loss------------\n")
    train_loader = validation_dataset.training_loader(batch_size)
    total_char_loss = 0
    total_word_loss = 0
    total_batches = 0
    for iteration, batch in enumerate(train_loader):
        # Each example in batch consists of
        # entity: The query term.
        # document: In the document JSON format from 'dictionary'.
        # targets: A list of ints the same length as the number of sentences
        # in the document.
        document_lengths = torch.Tensor([len(ex["sentences"])
                                         for ex in batch]).long()
        max_doc_length = torch.max(document_lengths)

        document_encodings = []
        terms = []
        for i, example in tqdm(list(enumerate(batch))):
            # Compute document representations for each document in 'batch'.
            # Encode the sentences to vector space before inference.
            terms.append(example["entity"])
            sentences = example["sentences"]
            encoded_sentences = validation_dataset.vectorize_sentences(sentences)
            document_tensor = get_document_tensor(encoded_sentences)
            document_tensor = document_tensor.view(-1, )

            # Each row will be a separate document.
            document_encodings.append(document_tensor)

        max_word_length = max(document_encodings, key=lambda x: x.size(0)).size(0)
        batch_document_text = torch.zeros(len(batch), max_word_length).long().to(device)
        for i, encoded_document in enumerate(document_encodings):
            batch_document_text[i][: len(encoded_document)] = encoded_document

        print("Char-level RNN training:")
        char_bptt_limit = 2
        for i, term in tqdm(enumerate(terms)):
            term_loss = 0
            for j in range(len(term) - char_bptt_limit - 1):
                input = term[j: j + char_bptt_limit]
                target = model.line_to_tensor(term[j + 1: j + 1 + char_bptt_limit]).long()
                predictions = model.char_level_forward(input)
                loss = cross_entropy(predictions, target)
                term_loss += loss.data.item()

            total_char_loss += (term_loss / len(term))

        for i in tqdm(range(max_doc_length)):
            input = batch_document_text[:, i: i + bptt_limit]
            target = batch_document_text[:, i + 1: i + 1 + bptt_limit]

            # Each batch contributes 'batch_size' predictions per sentence.
            # Shape: (batch_size,)
            predictions = model.word_level_forward(input)
            loss = cross_entropy(predictions, target)

            total_word_loss += loss.data.item()

        total_batches += 1

    return {
        "word": total_word_loss / total_batches,
        "character": total_char_loss / total_batches
    }


def init_dictionary(train_path, min_token_count):
    """
    Constructs a dictionary from Semantic Scholar JSONs found in 'train_path'.
    :param train_path: file path
        The path to the JSON documents meant for training / validation.
    :param min_token_count:
        The minimum number of times a word has to occur to be included.
    :return: A dictionary of training and development data.
    """
    all_training_examples = os.listdir(train_path)

    tokens = []
    for file in tqdm(all_training_examples):
        file_path = os.path.join(train_path, file)
        tokens += extract_tokens_from_json(file_path)

    # Map words to the number of times they occur in the dictionary.
    word_frequencies = dict(Counter(tokens))

    # Sieve the dictionary by excluding all words that appear fewer
    # than min_token_count times.
    vocabulary = set([w for w, f in word_frequencies.items()
                      if f >= min_token_count])

    # Construct the dictionary with the given vocabulary.
    dictionary = Dictionary(vocabulary)

    return dictionary


def get_document_tensor(sentences):
    max_length = len(max(sentences, key=lambda x: len(x)))
    sentences_tensor = torch.zeros(len(sentences), max_length).long()
    for j, sentence in enumerate(sentences):
        sentences_tensor[j, :len(sentence)] = sentence

    return sentences_tensor


def print_progress_in_place(*args):
    print("\r", *args, end="")
    sys.stdout.flush()


def save_model(model, save_dir, save_name):
    """
    Save a model to the disk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_weights = model.state_dict()
    save_path = os.path.join(save_dir, save_name)
    torch.save(model_weights, save_path)


if __name__ == "__main__":
    main()
