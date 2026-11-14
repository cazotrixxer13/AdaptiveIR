
import json
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_required_documents(dataset_name, document_ids):
    corpus_path = (
        Path("data/processed") / dataset_name / "corpus.jsonl"
    )

    documents = {}

    with corpus_path.open("r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            doc_id = doc["doc_id"]

            if doc_id in document_ids:
                documents[doc_id] = doc["retrieval_text"]

    return documents


def prepare_training_pairs(dataset_name):
    split_dir = Path("data/splits") / dataset_name

    queries = load_json(split_dir / "queries_retriever_train.json")
    qrels = load_json(split_dir / "qrels_retriever_train.json")

    positive_doc_ids = {
        doc_id for judged_docs in qrels.values() for doc_id, score in judged_docs.items() if score > 0
    }

    documents = load_required_documents(dataset_name, positive_doc_ids)

    output_dir = Path("data/training") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / "retriever_pairs.jsonl"
    model_path = output_dir / "retriever_model_inputs.jsonl"

    num_pairs = 0
    usable_queries = 0
    skipped_empty_docs = 0
    skipped_empty_queries = 0
    queries_without_positives = 0

    with (pairs_path.open("w", encoding="utf-8") as pairs_file, model_path.open("w", encoding="utf-8") as model_file):
        for query_id, judged_docs in sorted(qrels.items()):
            query_text = queries[query_id].strip()

            if not query_text:
                skipped_empty_queries += 1
                continue

            query_pair_count = 0

            for doc_id, score in sorted(judged_docs.items()):
                if score <= 0:
                    continue

                doc_text = documents[doc_id].strip()

                if not doc_text:
                    skipped_empty_docs += 1
                    continue

                pair = {
                    "dataset": dataset_name,
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "anchor": query_text,
                    "positive": doc_text,
                    "relevance_score": score
                }

                model_input = {"anchor": query_text,"positive": doc_text}

                pairs_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                model_file.write(json.dumps(model_input, ensure_ascii=False) + "\n")

                query_pair_count += 1
                num_pairs += 1

            if query_pair_count > 0:
                usable_queries += 1
            else:
                queries_without_positives += 1

    print(f"\n{dataset_name.upper()} TRAINING PAIRS")
    print(f"Retriever training queries: {len(queries)}")
    print(f"Queries with usable pairs: {usable_queries}")
    print(f"Generated positive pairs: {num_pairs}")
    print(f"Skipped empty positive documents: {skipped_empty_docs}")
    print(f"Skipped empty queries: {skipped_empty_queries}")
    print(f"Queries without usable positives: {queries_without_positives}")


def main():
    for dataset_name in ["fiqa", "scifact"]:
        prepare_training_pairs(dataset_name)


if __name__ == "__main__":
    main()
