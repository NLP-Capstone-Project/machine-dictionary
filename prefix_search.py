import argparse
import collections
import dill
import json
import os
import re
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Dictionary import UMLS


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--data-path", type=str,
                        help="List to semantic scholar parsed PDFs.")
    parser.add_argument("--save-dir", type=str,
                        help="Path to save relevant examples.")
    parser.add_argument("--definitions-path", type=str,
                        help="Path to the UMLS MRDEF.RRF file.")
    parser.add_argument("--synonyms-path", type=str,
                        help="Path to the UMLS MRCONSO.RRF file.")
    parser.add_argument("--common-english-path", type=str,
                        default=os.path.join(project_root, "google-10000-english.txt"),
                        help="File containing common english words that don't"
                             "need defining.")
    parser.add_argument("--pickled-umls", type=str,
                        default=os.path.join(project_root, "umls.pkl"),
                        help="A pickled set of UMLS entity-definition mappings.")

    args = parser.parse_args()

    with open(args.common_english_path, 'r') as f:
        english_words = set([word.strip() for word in f.readlines()])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Generate UMLS definitions.
    if not os.path.exists(args.pickled_umls):
        umls = UMLS(args.definitions_path, args.synonyms_path)
        umls.generate_all_definitions()

        # If any of the aliases is included in the top 10,000
        # most common english words, it's not interesting enough to
        # define.
        #
        # Allow special-cased words to break through (ex. cAMP is
        # not the same as camp).
        #
        # Disallow words that have no letters.
        filtered_mappings = {}
        print("Filtering UMLS terms based on Google's top 10K words:")
        for definition, aliases in tqdm(umls.definition_mappings.items()):
            keep = True
            for term in aliases:
                term_lowered = term[0].lower() + term[1:]
                discard = term_lowered in english_words or not any(let.isalpha() for let in term_lowered)
                discard = discard or '/' in term  # Disallow units of measure.
                if discard:
                    keep = False
                    break
            if keep:
                filtered_mappings[definition] = aliases

        with open(args.pickled_umls, 'wb') as f:
            dill.dump(filtered_mappings, f, protocol=dill.HIGHEST_PROTOCOL)

        print(len(filtered_mappings), "total definitions kept!")
    else:
        with open(args.pickled_umls, 'rb') as f:
            filtered_mappings = dill.load(f)

    # For each parsed document / definition pair, check if the aliases are
    # contained in the document.
    if not os.path.exists(args.save_dir):
        raise ValueError("Parsed directory invalid.")

    print("Beginning comparisons:")
    for document in os.listdir(args.data_path):
        document_path = os.path.join(args.data_path, document)
        with open(document_path, 'r') as f:
            document_json = json.load(f)

        # Make document, definition mappings be contained in a folder
        # specific to the document for easy analysis.
        formatted_title = re.sub(r'[^a-zA-Z0-9]', '_', document_json["title"])
        document_save_dir = os.path.join(args.save_dir, formatted_title)
        if os.path.exists(document_save_dir):
            continue

        hashed_title = str(abs(hash(formatted_title)) % (10 ** 15))
        try:
            os.mkdir(document_save_dir)
        except OSError as exc:
            if exc.errno == 36:
                # Take the first five words and call it a day.
                title_prefix_words = document_json["title"].split()[:5]
                title_prefix = '_'.join([re.sub(r'[^a-zA-Z0-9]', '_', word)
                                        for word in title_prefix_words])
                document_save_dir = os.path.join(args.save_dir,
                                                 title_prefix + hashed_title)
                if os.path.exists(document_save_dir):
                    continue
                os.mkdir(document_save_dir)
            else:
                continue

        translator = str.maketrans('', '', ".,:;'\"")
        split_sentences_no_punct = [sent.translate(translator).split()
                                    for sent in document_json["sentences"]]

        print("PROCESSING", document_json["title"])
        for (definition, aliases) in tqdm(filtered_mappings.items()):
            comparision = definition_document_comparison(document_json,
                                                         aliases,
                                                         definition,
                                                         split_sentences_no_punct)
            if comparision:
                comparision_path = hashed_title + re.sub(r'[^a-zA-Z0-9]', '_',
                                                         comparision["matching_term"])
                comparision_path = os.path.join(document_save_dir,
                                                comparision_path + ".json")
                with open(comparision_path, 'w') as f:
                    json.dump(comparision, f,
                              sort_keys=True,
                              ensure_ascii=False,
                              indent=4)

                progress_print = "Saved an example! {:20s} | for term {:15s} " \
                                 "| with prefix depth {:4f}"
                print(progress_print.format(document_json["title"],
                                            comparision["matching_term"],
                                            comparision["prefix_depth"]))


def definition_document_comparison(document_json, aliases, definition,
                                   split_sentences_no_punct,
                                   prefix_depth=0.15):
    """
    Given a parsed document, search within the first 'prefix_depth' of
    the sentences.
    """

    split_sentences_no_punct = split_sentences_no_punct[: int(len(split_sentences_no_punct) * prefix_depth)]
    for i, sentence in enumerate(split_sentences_no_punct):
        for term in aliases:
            if term in sentence:
                observation = collections.OrderedDict(
                    [("title", document_json["title"]),
                     ("definition", definition),
                     ("aliases", list(aliases)),
                     ("matching_term", term),
                     ("sentence_found", document_json["sentences"][i]),
                     ("prefix_depth", i / len(document_json["sentences"])),
                     ("sentences", document_json["sentences"])]
                )

                return observation

    return None


if __name__ == "__main__":
    main()
