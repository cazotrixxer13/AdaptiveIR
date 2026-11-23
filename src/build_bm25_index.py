
from pathlib import Path

import bm25s

from utils import DATASETS, read_jsonl, make_bm25_tokenizer


def load_documents(dataset_name):
    corpus_path = (Path("data/processed") / dataset_name / "corpus.jsonl")

    return [
        {"doc_id": row["doc_id"], "text": row["retrieval_text"]}
        for row in read_jsonl(corpus_path)
        if row["retrieval_text"].strip()
    ]


def build_bm25_index(dataset_name):
    documents = load_documents(dataset_name)
    texts = [doc["text"] for doc in documents]
    tokenizer = make_bm25_tokenizer()

    corpus_tokens = tokenizer.tokenize(
        texts,
        update_vocab=True,
        return_as="tuple",
        allow_empty=False
    )

    retriever = bm25s.BM25(
        method="lucene",
        k1=1.2,
        b=0.75,
        corpus=documents
    )

    retriever.index(corpus_tokens)

    output_dir = Path("data/bm25_index") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever.save(str(output_dir), corpus=documents)

    tokenizer.save_vocab(str(output_dir))
    tokenizer.save_stopwords(str(output_dir))

    print(f"\n{dataset_name.upper()} BM25 index built")
    print(f"Indexed documents: {len(documents)}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab_dict())}")


def main():
    for dataset_name in DATASETS:
        build_bm25_index(dataset_name)


if __name__ == "__main__":
    main()
