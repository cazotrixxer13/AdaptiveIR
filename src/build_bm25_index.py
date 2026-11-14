
import json
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenizer


def load_documents(dataset_name):
    corpus_path = (Path("data/processed") / dataset_name / "corpus.jsonl")
    documents = []

    with corpus_path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            if not row["retrieval_text"].strip():
                continue

            documents.append({
                "doc_id": row["doc_id"],
                "text": row["retrieval_text"]
            })

    return documents


def build_bm25_index(dataset_name):
    documents = load_documents(dataset_name)
    texts = [doc["text"] for doc in documents]
    tokenizer = Tokenizer(
        lower=True,
        splitter=r"(?u)\b\w+\b",
        stopwords=None,
        stemmer=None
    )

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
    for dataset_name in ["fiqa", "scifact"]:
        build_bm25_index(dataset_name)


if __name__ == "__main__":
    main()
