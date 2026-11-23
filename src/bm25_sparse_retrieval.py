import argparse

import bm25s

from utils import DATA_DIRECTORY, DATASETS, make_bm25_tokenizer


def load_saved_bm25(dataset_name: str):
    index_dir = DATA_DIRECTORY / "bm25_index" / dataset_name
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=True)
    tokenizer = make_bm25_tokenizer()
    tokenizer.load_vocab(str(index_dir))
    tokenizer.load_stopwords(str(index_dir))
    return retriever, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="fiqa")
    args = parser.parse_args()

    # retriever, tokenizer = load_saved_bm25(args.dataset)

if __name__ == "__main__":
    main()
