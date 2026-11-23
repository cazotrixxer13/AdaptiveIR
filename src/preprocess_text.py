
import re
from pathlib import Path

from utils import DATASETS, load_json, write_jsonl


def normalize_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def combine_title_text(title, text):
    title = normalize_text(title)
    text = normalize_text(text)

    return " ".join(part for part in [title, text] if part)


def preprocess_dataset(dataset_name):
    input_dir = Path("data/parsed") / dataset_name
    output_dir = Path("data/processed") / dataset_name

    corpus = load_json(input_dir / "corpus.json")
    queries = load_json(input_dir / "queries.json")

    empty_doc_ids = set()

    processed_docs = []

    for doc_id, doc in corpus.items():
        retrieval_text = combine_title_text(doc["title"], doc["text"])

        if not retrieval_text:
            empty_doc_ids.add(doc_id)

        processed_docs.append({"doc_id": doc_id, "retrieval_text": retrieval_text})

    write_jsonl(output_dir / "corpus.jsonl", processed_docs)
    write_jsonl(output_dir / "queries.jsonl", (
        {"query_id": query_id, "text": normalize_text(query_text)}
        for query_id, query_text in queries.items()
    ))

    print(f"\nProcessed {dataset_name.upper()}")
    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Empty documents: {len(empty_doc_ids)}")

    return empty_doc_ids


def check_empty_doc_qrels(dataset_name, empty_doc_ids):
    input_dir = Path("data/parsed") / dataset_name

    for qrels_file in sorted(input_dir.glob("qrels_*.json")):
        qrels = load_json(qrels_file)

        affected_queries = set()
        relevant_judgments = 0

        for query_id, docs in qrels.items():
            for doc_id, score in docs.items():
                if doc_id in empty_doc_ids and score > 0:
                    affected_queries.add(query_id)
                    relevant_judgments += 1

        print(
            f"{qrels_file.stem}: "
            f"{len(affected_queries)} affected queries, "
            f"{relevant_judgments} relevant judgments"
        )


def main():
    for dataset_name in DATASETS:
        empty_doc_ids = preprocess_dataset(dataset_name)
        check_empty_doc_qrels(dataset_name, empty_doc_ids)


if __name__ == "__main__":
    main()