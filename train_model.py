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
                                         args.hidden_size, args.batch_size,
                                         layers=1, dropout=args.dropout).to(device)

    logger.info(model)

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
                                      for paper_dir in os.listdir(args.bio_dir)]
            validation_dataset = UMLSCorpus(dictionary, extractor, umls)
            for valid_dir in validation_directories:
                validation_dataset.collect_all_data(valid_dir)

            print("Training on", len(paper_directories), "documents...")

            # Train the sequence tagger.
            for epoch in range(args.num_epochs):
                for paper_dir in paper_directories:
                    umls_dataset.collect_all_data(paper_dir)
                    train_tagger_epoch(model, umls_dataset, validation_dataset,
                                       args.batch_size, optimizer,
                                       args.save_dir, epoch, log_period=args.log_period)

            print()  # Printing in-place progress flushes standard out.
        except KeyboardInterrupt:
            print("\nStopped training early.")
            pass


def train_tagger_epoch(model, umls_dataset, validation_dataset,
                       batch_size, optimizer,
                       save_dir, epoch, log_period=5):
    """
    Train the tagger model for one epoch.
    """

    model.train()
    print("Training in progress:\n")
    train_loader = umls_dataset.training_loader(batch_size)
    for iteration, batch in enumerate(train_loader):
        # Each example in batch consists of
        # entity: The query term.
        # document: In the document JSON format from 'dictionary'.
        # targets: A list of ints the same length as the number of sentences
        # in the document.
        document_lengths = torch.Tensor([len(ex["document"]["sentences"])
                                         for ex in batch]).long()
        max_doc_length = torch.max(document_lengths)

        # Concatenate predictions and construct target tensor:
        # Dataset predictions are one tensor per document; fill the
        # predictions len(batch) elements at a time.
        all_targets = torch.zeros(len(batch), max_doc_length).long()
        for i, ex in enumerate(batch):
            # Jump to current target and encode a max_doc_length
            # vector with the proper hits.
            targets = torch.zeros(max_doc_length.item())
            targets[:len(ex["targets"])] = torch.Tensor(ex["targets"])
            all_targets[i] = targets

        all_targets = all_targets.contiguous().long().to(device)

        # Shape: (batch x max document length x bidirectional hidden)
        batch_hidden_states = torch.zeros(len(batch), max_doc_length,
                                          model.hidden_size * 2)
        batch_hidden_states = batch_hidden_states.float()

        # Shape: (batch x bidirectional hidden size)
        batch_document_reps = torch.Tensor(len(batch), model.hidden_size * 2)

        # Shape: (batch x bidirectional hidden size)
        batch_term_reps = torch.Tensor(len(batch), model.hidden_size * 2)

        document_lengths = document_lengths.to(device)
        batch_hidden_states = batch_hidden_states.to(device)
        batch_document_reps = batch_document_reps.to(device)
        batch_term_reps = batch_term_reps.to(device)

        print("Computing document representations:")
        for i, example in tqdm(list(enumerate(batch))):
            # Compute document representations for each document in 'batch'.
            # Encode the sentences to vector space before inference.
            document = example["document"]
            sentences = document["sentences"]
            encoded_sentences = umls_dataset.vectorize_sentences(sentences)
            document_tensor = get_document_tensor(encoded_sentences).to(device)

            # Compute term representations for each term in batch
            term = example["entity"]
            term_rep = model.term_representation(term)

            # Shapes: (document_length x hidden_size), (hidden_size,)
            # Set the global batch level hidden state and
            # document representation tensors.
            hiddens, doc_rep = model.document_representation(document_tensor)
            batch_hidden_states[i, :hiddens.size(0)] = hiddens
            batch_document_reps[i] = doc_rep
            batch_term_reps[i] = term_rep

        # For calculating novelty, we need a running summary over sentence
        # hidden states represented with
        #     s_j = sum_{i = 1}^{j - 1} h_i * P(y_j | h_i, s_i, d)
        sum_rep = torch.zeros(len(batch), model.hidden_size * 2).to(device)

        # Iterate over sentences and predict their BIO tag:
        print("Predictions on sentences:")
        batch_loss = 0
        for i in tqdm(range(max_doc_length)):
            # Column Shape: (Batch, hidden_size)
            # Each row is a sentence: sum across hiddens to get a sense
            current_sentence_hiddens = batch_hidden_states[:, i]
            hidden_state_sums = current_sentence_hiddens.sum(dim=1)
            inference_mask = (hidden_state_sums != 0).float()

            # Each batch contributes 'batch_size' predictions per sentence.
            # Shape: (batch_size,)
            predictions = model.forward(current_sentence_hiddens, i, sum_rep,
                                        document_lengths, batch_document_reps,
                                        batch_term_reps)

            predictions = predictions.squeeze() * inference_mask
            batch_predictions = torch.Tensor(predictions.size(0), 2)
            batch_predictions[:, 0] = 1 - predictions
            batch_predictions[:, 1] = predictions
            batch_targets = all_targets[:, i]

            optimizer.zero_grad()
            loss = cross_entropy(batch_predictions, batch_targets)
            loss.backward(retain_graph=True)
            optimizer.step()

            # Update summary representation.
            # Some predictions are zero given the mask; at this point it's okay
            # for some representations to get zeroes out as they won't
            # contribute to loss anyway.
            #
            # Unsqueeze at the first dimensions so that they match at the
            # zeroeth.
            # Shape: (batch size x bidirectional hidden size)
            sum_rep = predictions.unsqueeze(1) * current_sentence_hiddens

            batch_loss += loss.data

        print_progress_in_place("Batch #", iteration,
                                ", Loss for the batch:", batch_loss.data[0])
        print()

        if iteration % log_period == 0:
            # TODO: Validation Loss?
            evaluation = evaluate(model, validation_dataset,
                                  batch_size=batch_size)
            accuracy = evaluation["accuracy"]
            average_loss = evaluation["loss"] / evaluation["attempted"]
            save_name = "model_epoch{}_batch{}.ph".format(epoch, iteration)
            save_model(model, save_dir, save_name)


def evaluate(model, validation_dataset,
             batch_size=20):
    """
    Compute validation loss given the model.
    """

    model.eval()
    print("------------Computing validation loss------------\n")
    train_loader = validation_dataset.training_loader(batch_size)
    total_loss = 0
    total_correct = 0
    total_attempted = 0
    total_batches = 0
    for iteration, batch in enumerate(train_loader):
        # Each example in batch consists of
        # entity: The query term.
        # document: In the document JSON format from 'dictionary'.
        # targets: A list of ints the same length as the number of sentences
        # in the document.
        document_lengths = torch.Tensor([len(ex["document"]["sentences"])
                                         for ex in batch]).long()
        max_doc_length = torch.max(document_lengths)

        # Concatenate predictions and construct target tensor:
        # Dataset predictions are one tensor per document; fill the
        # predictions len(batch) elements at a time.
        all_targets = torch.zeros(len(batch), max_doc_length).long()
        for i, ex in enumerate(batch):
            # Jump to current target and encode a max_doc_length
            # vector with the proper hits.
            targets = torch.zeros(max_doc_length.item())
            targets[:len(ex["targets"])] = torch.Tensor(ex["targets"])
            all_targets[i] = targets

        all_targets = all_targets.contiguous().long().to(device)

        # Shape: (batch x max document length x bidirectional hidden)
        batch_hidden_states = torch.zeros(len(batch), max_doc_length,
                                          model.hidden_size * 2)
        batch_hidden_states = batch_hidden_states.float()

        # Shape: (batch x bidirectional hidden size)
        batch_document_reps = torch.Tensor(len(batch), model.hidden_size * 2)

        # Shape: (batch x bidirectional hidden size)
        batch_term_reps = torch.Tensor(len(batch), model.hidden_size * 2)

        document_lengths = document_lengths.to(device)
        batch_hidden_states = batch_hidden_states.to(device)
        batch_document_reps = batch_document_reps.to(device)
        batch_term_reps = batch_term_reps.to(device)

        print("Computing document representations:")
        for i, example in tqdm(list(enumerate(batch))):
            # Compute document representations for each document in 'batch'.
            # Encode the sentences to vector space before inference.
            document = example["document"]
            sentences = document["sentences"]
            encoded_sentences = validation_dataset.vectorize_sentences(sentences)
            document_tensor = get_document_tensor(encoded_sentences).to(device)

            # Compute term representations for each term in batch
            term = example["entity"]
            term_rep = model.term_representation(term)

            # Shapes: (document_length x hidden_size), (hidden_size,)
            # Set the global batch level hidden state and
            # document representation tensors.
            hiddens, doc_rep = model.document_representation(document_tensor)
            batch_hidden_states[i, :hiddens.size(0)] = hiddens
            batch_document_reps[i] = doc_rep
            batch_term_reps[i] = term_rep

        # For calculating novelty, we need a running summary over sentence
        # hidden states represented with
        #     s_j = sum_{i = 1}^{j - 1} h_i * P(y_j | h_i, s_i, d)
        sum_rep = torch.zeros(len(batch), model.hidden_size * 2).to(device)

        # Iterate over sentences and predict their BIO tag:
        print("Predictions on sentences:")
        for i in tqdm(range(max_doc_length)):
            # Column Shape: (Batch, hidden_size)
            # Each row is a sentence: sum across hiddens to get a sense
            current_sentence_hiddens = batch_hidden_states[:, i]
            hidden_state_sums = current_sentence_hiddens.sum(dim=1)
            inference_mask = (hidden_state_sums != 0).float()

            # Each batch contributes 'batch_size' predictions per sentence.
            # Shape: (batch_size,)
            predictions = model.forward(current_sentence_hiddens, i, sum_rep,
                                        document_lengths, batch_document_reps,
                                        batch_term_reps)

            predictions = predictions.squeeze() * inference_mask
            batch_predictions = torch.Tensor(predictions.size(0), 2)
            batch_predictions[:, 0] = 1 - predictions
            batch_predictions[:, 1] = predictions
            batch_targets = all_targets[:, i]

            loss = cross_entropy(batch_predictions, batch_targets)
            sum_rep = predictions.unsqueeze(1) * current_sentence_hiddens

            # Compute accuracy:
            threshold = torch.zeros_like(batch_targets)
            threshold.fill_(0.4999)
            predicted_indices = (predictions > threshold).long()
            compare = (predicted_indices == batch_targets).long()
            correct = compare.sum()
            attempted = batch_size

            total_loss += loss.data.item()
            total_correct += correct
            total_attempted += attempted

        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "accuracy": total_correct / total_attempted
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
    serialization_dictionary = {
        "model_type": model.__class__.__name__,
        "model_weights": model_weights,
        "init_arguments": model.init_arguments,
        "global_step": model.global_step
    }

    save_path = os.path.join(save_dir, save_name)
    torch.save(serialization_dictionary, save_path)


if __name__ == "__main__":
    main()
