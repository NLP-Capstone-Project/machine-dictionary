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
from Dictionary import Dictionary, UMLSCorpus,\
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

    if not os.path.exists(args.vocabulary_path):
        print("Restricting vocabulary based on min token count",
              args.min_token_count)

        print("Constructing Dictionary:")
        dictionary = init_dictionary(args.train_path, args.parsed_dir, args.min_token_count)
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
                                         layers=1, dropout=args.dropout)

    if args.cuda:
        model.cuda()

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

            # Generate examples from parsed directory if not present.
            if not os.path.exists(args.bio_dir):
                os.mkdir(args.bio_dir)
                umls_dataset.generate_all_data(args.bio_dir, args.sparsed_dir)
            else:
                umls_dataset.collect_all_data(args.bio_dir)

            # Train the sequence tagger.
            train_tagger_epoch(model, umls_dataset, args.batch_size,
                               optimizer, args.cuda)

            print()  # Printing in-place progress flushes standard out.
        except KeyboardInterrupt:
            print("\nStopped training early.")
            pass
    else:
        try:
            train_epoch(model, dictionary, args.batch_size, args.bptt_limit, optimizer,
                        args.cuda, args.model_type)
            print()  # Printing in-place progress flushes standard out.
        except KeyboardInterrupt:
            print("\nStopped training early.")
            pass
        # Evaluation: Calculating perplexity.
        perplexity = evaluate_perplexity(model, dictionary, args.batch_size,
                                         args.bptt_limit, args.cuda)

        print("\nFinal perplexity for validation: {.4f}", perplexity)


def train_tagger_epoch(model, umls_dataset, batch_size, optimizer, cuda):
    """
    Train the tagger model for one epoch.
    """

    # Set model to training mode (activates dropout and other things).
    model.train()
    print("Training in progress:\n")

    train_loader = umls_dataset.training_loader(batch_size)

    for iteration, batch in enumerate(train_loader):
        # Each example in batch consists of
        # entity: The query term.
        # document: In the document JSON format from 'dictionary'.
        # targets: A list of ints the same length as the number of sentences
        # in the document.
        document_lengths = torch.LongTensor([len(ex["document"]["sentences"]) for ex in batch])
        max_doc_length = torch.max(document_lengths)

        # Concatenate predictions and construct target tensor:
        # Dataset predictions are one tensor per document; fill the
        # predictions len(batch) elements at a time.
        #
        # len(batch) is used instead of batch_size to allow the last
        # mis-aligned batch to be trained on.
        all_predictions = Variable(torch.zeros(len(batch) * max_doc_length).float())
        all_targets = Variable(torch.zeros(len(batch), max_doc_length))
        for i, ex in enumerate(batch):
            # Jump to current target and encode a max_doc_length
            # vector with the proper hits.
            targets = [0] * max_doc_length
            targets[:len(ex["targets"])] = ex["targets"]
            targets = Variable(torch.LongTensor(targets))
            all_targets[i] = targets

        # Concatenate the columns so that every ith batch_size chunk corresponds
        # to the ith sentence of the examples in the batch.
        # 1. Transpose to get batches row-wise.
        # 2. Contiguous to meld memory together.
        # 3. Reshape and squeeze into expected shape for targets.
        # 4. Targets need to be Longs.
        all_targets = all_targets.t().contiguous().view(-1, 1).squeeze().long()

        # Shape: (batch x max document length x bidirectional hidden)
        batch_hidden_states = torch.zeros(len(batch), max_doc_length, model.hidden_size * 2)
        batch_hidden_states = batch_hidden_states.float()

        # Shape: (batch x bidirectional hidden size)
        batch_document_reps = torch.FloatTensor(len(batch), model.hidden_size * 2)

        # Shape: (batch x bidirectional hidden size)
        batch_term_reps = torch.FloatTensor(len(batch), model.hidden_size * 2)

        if cuda:
            document_lengths = document_lengths.cuda()
            batch_hidden_states = batch_hidden_states.cuda()
            batch_document_reps = batch_document_reps.cuda()
            batch_term_reps = batch_term_reps.cuda()

        batch_hidden_states = Variable(batch_hidden_states)
        batch_document_reps = Variable(batch_document_reps)
        batch_term_reps = Variable(batch_term_reps)

        print("Computing document representations:")
        for i, example in tqdm(list(enumerate(batch))):
            # Compute document representations for each document in 'batch'.
            # Encode the sentences to vector space before inference.
            document = example["document"]
            encoded_sentences = umls_dataset.vectorize_sentences(document["sentences"])
            document_tensor = get_document_tensor(encoded_sentences)

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
        sum_rep = Variable(torch.zeros(len(batch), model.hidden_size * 2))

        # Iterate over sentences and predict their BIO tag:
        print("Predictions on sentences:")
        for i in tqdm(range(max_doc_length)):
            # Column Shape: (Batch, hidden_size)
            # Each row is a sentence: sum across hiddens to get a sense
            current_sentence_hiddens = batch_hidden_states[:, i]
            hidden_state_sums = current_sentence_hiddens.sum(dim=1)
            inference_mask = (hidden_state_sums != 0).float()

            # To compute loss, make one long predictions tensor where the
            # result is the concatenation of all masked prediction vectors
            # compared against the concatenation of all labels.
            #
            # Each batch contributes 'batch_size' predictions per sentence.
            # Shape: (batch_size,)
            predictions = model.forward(current_sentence_hiddens, i, sum_rep,
                                        document_lengths, batch_document_reps, batch_term_reps)

            predictions = predictions.squeeze() * inference_mask
            place = i * len(batch)  # Jump to current prediction index and map.
            all_predictions[place: place + len(batch)] = predictions

            # Update summary representation.
            # Some predictions are zero given the mask; at this point it's okay
            # for some representations to get zeroes out as they won't
            # contribute to loss anyway.
            #
            # Unsqueeze at the first dimensions so that they match at the
            # zeroeth.
            # Shape: (batch size x bidirectional hidden size)
            sum_rep = predictions.unsqueeze(1) * current_sentence_hiddens

        # Construct final predictions: Cross entropy requires all probabilities
        # for each class. Currently, all_predictions contains
        # batch size x max doc length probabilities. We construct the complimentary
        # probabilities 1 - all_predictions.
        #
        # Note that predictions are based on whether a sentence belongs, which
        # is label 1.
        final_predictions = Variable(torch.FloatTensor(all_predictions.size(0), 2))
        final_predictions[:, 0] = 1 - all_predictions
        final_predictions[:, 1] = all_predictions

        import pdb
        pdb.set_trace()

        optimizer.zero_grad()
        loss = nll_loss(final_predictions, all_targets)
        loss.backward()
        optimizer.step()

        print_progress_in_place("BATCH", iteration,
                                "LOSS:", loss.data[0])
        print()


def train_epoch(model, dictionary, batch_size, bptt_limit, optimizer, cuda, model_type):
    """
    Train the model for one epoch.
    """

    # Set model to training mode (activates dropout and other things).
    model.train()
    print("Training in progress:")
    for i, document in enumerate(dictionary.training):
        # Incorporation of time requires feeding in by one word at
        # a time.
        #
        # Iterate through the words of the document, calculating loss between
        # the current word and the next, from first to penultimate.
        for j, section in enumerate(document["sections"]):
            loss = 0
            hidden = model.init_hidden()

            # Training at the word level allows flexibility in inference.
            for k in range(section.size(0) - 1):
                current_word = word_vector_from_seq(section, k)
                next_word = word_vector_from_seq(section, k + 1)

                if cuda:
                    current_word = current_word.cuda()
                    next_word = next_word.cuda()

                output, hidden = model(Variable(current_word), hidden)

                # Calculate loss between the next word and what was anticipated.
                loss += cross_entropy(output.view(1, -1), Variable(next_word))

                # Perform backpropagation and update parameters.
                #
                # Detaches hidden state history to prevent bp all the way
                # back to the start of the section.
                if (k + 1) % bptt_limit == 0:
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Print progress
                    print_progress_in_place("Document #:", i,
                                            "Section:", j,
                                            "Word:", k,
                                            "Normalized BPTT Loss:",
                                            loss.data[0] / bptt_limit)

                    loss = 0
                    if type(hidden) is tuple:
                        hidden = (Variable(hidden[0].data), Variable(hidden[1].data))
                    else:
                        hidden = Variable(hidden.data)


def evaluate_perplexity(model, dictionary, batch_size, bptt_limit, cuda):
    """
    Calculate perplexity of the trained model for the given dictionary.
    """

    M = 0  # Word count.
    log_prob_sum = 0  # Log Probability

    # Set model to evaluation mode (deactivates dropout).
    model.eval()
    print("Evaluation in progress: Perplexity")
    for i, document in enumerate(dictionary.validation):
        # Iterate through the words of the document, calculating log
        # probability of the next word given the history at the time.
        for j, section in enumerate(document["sections"]):
            hidden = model.init_hidden()

            # Training at the word level allows flexibility in inference.
            for k in range(section.size(0) - 1):
                current_word = word_vector_from_seq(section, k)
                next_word = word_vector_from_seq(section, k + 1)

                if cuda:
                    current_word = current_word.cuda()
                    next_word = next_word.cuda()

                output, hidden = model(Variable(current_word), hidden)

                # Calculate probability of the next word given the model
                # in log space.
                # Reshape to (vocab size x 1) and perform log softmax over
                # the first dimension.
                prediction_probabilities = log_softmax(output.view(-1, 1), 0)

                # Extract next word's probability and update.
                prob_next_word = prediction_probabilities[next_word[0]]
                log_prob_sum += prob_next_word.data[0]
                M += 1

                # Detaches hidden state history at the same rate that is
                # done in training.
                if (k + 1) % bptt_limit == 0:
                    # Print progress
                    print_progress_in_place("Document #:", i,
                                            "Section:", j,
                                            "Word:", k,
                                            "M:", M,
                                            "Log Prob Sum:", log_prob_sum,
                                            "Normalized Perplexity thus far:",
                                            2 ** (-(log_prob_sum / M)))

                    if type(hidden) is tuple:
                        hidden = (Variable(hidden[0].data), Variable(hidden[1].data))
                    else:
                        hidden = Variable(hidden.data)

    # Final model perplexity given the dictionary.
    return 2 ** (-(log_prob_sum / M))


def init_dictionary(train_path, parsed_path, min_token_count):
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

    print("Building dictionary from Semantic Scholar JSON files:")
    for file in tqdm(all_training_examples):
        # dictionary expects a full file path.
        dictionary.save_processed_document(os.path.join(train_path, file),
                                           os.path.join(parsed_path, file))

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


if __name__ == "__main__":
    main()
