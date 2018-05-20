import argparse
import collections
import dill
import json
import os
import re
import sys
import time


from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Dictionary import load_embeddings, Extractor


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--documents-path", type=str,
                        help="Path to directory of directories")
    parser.add_argument("--save-dir", type=str,
                        help="Path to save relevant examples.")
    parser.add_argument("--vocabulary-path", type=str,
                        help="Path to the vocabulary")
    parser.add_argument("--glove-path", type=str,
                        help="Path to the glove vectors")

    args = parser.parse_args()

    # Reading in the vocabulary and glove vectors
    print("Parsing existing vocabulary")
    embedding_matrix, vocabulary = load_embeddings(args.glove_path, args.vocabulary_path)

    print("Creating saved directory")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print("Reading files and Extracting")
    all_paper_directories = os.listdir(args.documents_path)
    extractor = Extractor(embedding_matrix, vocabulary)
    for directory in tqdm(all_paper_directories):
        directory_path = os.path.join(args.documents_path, directory)
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            prefix_document = json.load(open(file_path, 'r'))



if __name__ == "__main__":
    main()
