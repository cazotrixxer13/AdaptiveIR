
import re
import json
from pathlib import Path


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
    output_dir.mkdir(parents=True, exist_ok=True)

    with (input_dir / "corpus.json").open("r", encoding="utf-8") as file:
        corpus = json.load(file)

    with (input_dir / "queries.json").open("r", encoding="utf-8") as file:
        queries = json.load(file)

    empty_doc_ids = set()

    with (output_dir / "corpus.jsonl").open("w", encoding="utf-8") as file:
        for doc_id, doc in corpus.items():
            retrieval_text = combine_title_text(doc["title"], doc["text"])

            if not retrieval_text:
                empty_doc_ids.add(doc_id)

            processed_doc = {
                "doc_id": doc_id,
                "retrieval_text": retrieval_text
            }

            file.write(json.dumps(processed_doc, ensure_ascii=False) + "\n")

    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as file:
        for query_id, query_text in queries.items():
            clean_query = normalize_text(query_text)

            processed_query = {
                "query_id": query_id,
                "text": clean_query
            }

            file.write(json.dumps(processed_query, ensure_ascii=False) + "\n")

    print(f"\nProcessed {dataset_name.upper()}")
    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Empty documents: {len(empty_doc_ids)}")

    return empty_doc_ids


def check_empty_doc_qrels(dataset_name, empty_doc_ids):
    input_dir = Path("data/parsed") / dataset_name

    for qrels_file in sorted(input_dir.glob("qrels_*.json")):
        with qrels_file.open("r", encoding="utf-8") as file:
            qrels = json.load(file)

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
    for dataset_name in ["fiqa", "scifact"]:
        empty_doc_ids = preprocess_dataset(dataset_name)
        check_empty_doc_qrels(dataset_name, empty_doc_ids)


if __name__ == "__main__":
    main()