
from datasets import load_dataset
import json
from pathlib import Path

def load_beir_dataset(dataset_name):
    corpus_raw = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    queries_raw = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
    qrels_raw = load_dataset(f"BeIR/{dataset_name}-qrels")

    corpus = {}

    for row in corpus_raw:
        doc_id = str(row["_id"])
        corpus[doc_id] = {"title": row["title"], "text": row["text"]}

    queries = {}

    for row in queries_raw:
        query_id = str(row["_id"])
        queries[query_id] = row["text"]

    qrels = {}

    for split_name, split_data in qrels_raw.items():
        qrels[split_name] = {}

        for row in split_data:
            query_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(row["score"])

            if query_id not in qrels[split_name]:
                qrels[split_name][query_id] = {}

            qrels[split_name][query_id][doc_id] = score

    return corpus, queries, qrels


def validate_dataset(corpus, queries, qrels):
    document_ids = set(corpus.keys())
    query_ids = set(queries.keys())

    for split, judgments in qrels.items():
        for query_id, relevant_docs in judgments.items():

            if query_id not in query_ids:
                raise ValueError(f"Query {query_id} in {split} does not exist")

            for doc_id in relevant_docs:
                if doc_id not in document_ids:
                    raise ValueError(f"Document {doc_id} in {split} does not exist")

    empty_docs = sum(not (doc["title"].strip() or doc["text"].strip()) for doc in corpus.values())

    print(f"Documents with no content: {empty_docs}")
    print("Dataset validation passed!")


def save_parsed_dataset(dataset_name, corpus, queries, qrels):
    output_dir = Path("data/parsed") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(filename, data):
        with (output_dir / filename).open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

    save_json("corpus.json", corpus)
    save_json("queries.json", queries)

    for split, judgments in qrels.items():
        save_json(f"qrels_{split}.json", judgments)


def main():
    for dataset_name in ["fiqa", "scifact"]:
        print(f"\nParsing {dataset_name.upper()}")

        corpus, queries, qrels = load_beir_dataset(dataset_name)
        validate_dataset(corpus, queries, qrels)
        save_parsed_dataset(dataset_name, corpus, queries, qrels)

        print(f"Documents: {len(corpus)}")
        print(f"Queries: {len(queries)}")

        for split, judgments in qrels.items():
            num_qrels = sum(len(docs) for docs in judgments.values())

            print(
                f"{split}: {len(judgments)} queries, "
                f"{num_qrels} relevance judgments"
            )


if __name__ == "__main__":
    main()
