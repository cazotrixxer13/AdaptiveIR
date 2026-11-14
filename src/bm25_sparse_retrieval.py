import argparse
import json
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_saved_bm25(dataset_name: str):
    index_dir = (PROJECT_ROOT / "data" / "bm25_index" / dataset_name)
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=True)
    tokenizer = Tokenizer(
        lower=True,
        splitter=r"(?u)\b\w+\b",
        stopwords=None,
        stemmer=None,
    )
    tokenizer.load_vocab(str(index_dir))
    tokenizer.load_stopwords(str(index_dir))
    return retriever, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fiqa", "scifact"], default="fiqa")
    args = parser.parse_args()

    # retriever, tokenizer = load_saved_bm25(args.dataset)

if __name__ == "__main__":
    main()
