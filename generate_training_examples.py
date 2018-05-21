import argparse
import collections
import json
import os
import sys


from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--documents-path", type=str,
                        help="Path to directory of directories")
    parser.add_argument("--save-dir", type=str,
                        help="Path to save relevant examples.")

    args = parser.parse_args()

    print("Creating saved directory")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print("Reading files and Extracting")
    all_paper_directories = os.listdir(args.documents_path)
    for directory in tqdm(all_paper_directories):
        directory_path = os.path.join(args.documents_path, directory)
        files = os.listdir(directory_path)

        document_save_dir = os.path.join(args.save_dir, directory)
        if not os.path.exists(document_save_dir):
            os.mkdir(document_save_dir)

        for file in files:
            file_path = os.path.join(directory_path, file)
            ranked_document = json.load(open(file_path, 'r'))

            entity = ""

            output = collections.OrderedDict(
                [("title", ranked_document["title"]),
                 ("entity", entity),
                 ("definition", ranked_document["definition"]),
                 ("sentences", ranked_document["sentences"]),
                 ("target", ranked_document["target"])]
            )

            file_save_path = os.path.join(document_save_dir, file)
            with open(file_save_path, 'w') as f:
                json.dump(output, f,
                          sort_keys=True,
                          ensure_ascii=False,
                          indent=4)


if __name__ == "__main__":
    main()
