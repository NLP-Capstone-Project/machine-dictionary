import argparse
from collections import Counter
import dill
import json
import os
from tqdm import tqdm
import re
import sys

from nltk.tokenize import word_tokenize

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Dictionary import Dictionary, UMLSCorpus, UMLS, Extractor


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--train-path", type=str,
                        default=os.path.join(
                            project_root, "data", "train"),
                        help="Path to the Semantic Scholar training data.")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(
                            project_root, "data", "papers"),
                        help="Path to the directory containing the data.")
    parser.add_argument("--built-dictionary-path", type=str,
                        default=os.path.join(
                            project_root, "dictionary.pkl"),
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
                        help=("Path to save the collected data. "
                              "Houses parsed and BIO dirs. "),
                        default=os.path.join(
                            project_root, "data"))
    parser.add_argument("--min-token-count", type=int, default=5,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--skip-parsing", type=bool, default=False,
                        help="If true, skips ahead to BIO-tagging.")
    parser.add_argument("--skip-bio", type=bool, default=False,
                        help="If true, skips BIO-tagging")
    parser.add_argument("--vocabulary-path", type=str, default="vocabulary.txt",
                        help="If skip-parsing is true, loads in the vocabulary"
                             "this file.")
    args = parser.parse_args()

    # Set the groundwork for constructing a UMLS corpus.
    #
    # Set up explicit dir just for parsed documents and
    # for the BIO-tagged annotated versions.
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    parsed_dir = os.path.join(args.save_dir, "parsed")
    if not os.path.exists(parsed_dir):
        os.mkdir(parsed_dir)

    bio_dir = os.path.join(args.save_dir, "BIO")
    if not os.path.exists(bio_dir):
        os.mkdir(bio_dir)

    print("Restricting vocabulary based on min token count",
          args.min_token_count)

    # Construct and pickle the dictionary to prevent the need to recalculate
    # mappings and vocab size.
    if not args.skip_parsing:
        print("Collecting Semantic Scholar JSONs:")
        paper_directories = [os.path.join(args.data_dir, paper_dir)
                             for paper_dir in os.listdir(args.data_dir)]
        vocabulary = sieve_vocabulary(paper_directories, args.min_token_count)
        dictionary = Dictionary(vocabulary)
        pickled_dictionary = open(args.built_dictionary_path, 'wb')
        dill.dump(dictionary, pickled_dictionary)
    else:
        with open(args.vocabulary_path, 'r') as f:
            vocabulary = set([word.strip for word in f.readlines()])
        dictionary = Dictionary(vocabulary)

    vocab_size = len(dictionary)
    print("Vocabulary Size:", vocab_size)

    # Extract references from UMLS and create annotated data.
    if not args.skip_bio:
        umls = UMLS(args.definitions_path, args.synonyms_path)
        print("Generating UMLS Definitions:")
        umls.generate_all_definitions()
        extractor = Extractor(args.rouge_threshold, args.rouge_type)
        umls_dataset = UMLSCorpus(dictionary, extractor, umls)
        try:
            print("BIO-Tagging parsed documents.")
            umls_dataset.generate_all_data(bio_dir, parsed_dir)
            print()  # Printing in-place progress flushes standard out.
        except KeyboardInterrupt:
            print("\nStopping BIO tagging early.")
            sys.exit()


def sieve_vocabulary(paper_directories, min_token_count):
    """
    Parses and saves Semantic Scholar JSONs found in 'train_path'.
    :param paper_directories: file path
        The path to the paper subdirectories meant for training / validation.
    :param min_token_count:
        The minimum number of times a word has to occur to be included.
    """
    save_path = "vocabulary_" + str(min_token_count) + ".txt"
    print("Creating vocabulary from JSONs:")
    entities = set()
    counter = Counter()
    try:
        for paper_dir in tqdm(paper_directories):
            examples = os.listdir(paper_dir)
            for i, file in enumerate(examples):
                file_path = os.path.join(paper_dir, file)
                if i == 0:
                    file_tokens, entity = extract_tokens_from_BIO_json(file_path,
                                                                       entity_only=False)
                else:
                    file_tokens, entity = extract_tokens_from_BIO_json(file_path,
                                                                       entity_only=True)

                for token in file_tokens:
                    counter[token] += 1
                entities.add(entity)

    except KeyboardInterrupt:
        print("\n\nStopping Vocab Search Early.\n")

    # Map words to the number of times they occur in the corpus.
    word_frequencies = dict(counter)

    # Sieve the dictionary by excluding all words that appear fewer
    # than min_token_count times.
    #
    # Also exclude words that have no letters.
    vocabulary = set([w for w, f in word_frequencies.items()
                      if f >= min_token_count and
                      re.match(r'[a-zA-Z]', w)])
    with open(save_path, 'w') as f:
        for word in vocabulary:
            print(word, file=f)

    vocabulary = vocabulary.union(entities)  # Enforce that entities are always present.
    return vocabulary


def extract_tokens_from_BIO_json(path, entity_only=False):
    """
    Tokenizes a JSON article and returns a list
    of its tokens.

    If a file does not have "title" and "sections" field, this
    function returns an empty list.
    :param path: The path to a training document.
    """
    parsed_document = json.load(open(path, 'r'))

    # Collect the content sections.
    entity = parsed_document["entity"]
    sentences = parsed_document["sentences"]
    if not sentences:
        return [], None

    tokens = []

    if entity_only:
        return tokens, entity

    for sentence in sentences:
        tokens += word_tokenize(sentence)

    return tokens, entity


def print_progress_in_place(*args):
    print("\r", *args, end="")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
