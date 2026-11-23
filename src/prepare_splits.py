
import random
from pathlib import Path

from utils import DATASETS, load_json, read_jsonl, save_json

SEED = 6
SCIFACT_VALIDATION_RATIO = 0.20
ROUTER_TRAIN_RATIO = 0.60


def load_processed_queries(dataset_name):
    path = Path("data/processed") / dataset_name / "queries.jsonl"

    return {row["query_id"]: row["text"] for row in read_jsonl(path)}


def load_qrels(dataset_name):
    directory = Path("data/parsed") / dataset_name
    qrels = {}

    for path in directory.glob("qrels_*.json"):
        split = path.stem.removeprefix("qrels_")
        qrels[split] = load_json(path)

    return qrels


def split_query_ids(query_ids, holdout_ratio, seed):
    ids = sorted(query_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    holdout_count = round(len(ids) * holdout_ratio)
    holdout_ids = set(ids[:holdout_count])
    remaining_ids = set(ids[holdout_count:])

    return remaining_ids, holdout_ids


def select_qrels(qrels, query_ids):
    return {query_id: qrels[query_id] for query_id in sorted(query_ids)}


def prepare_dataset(dataset_name):
    queries = load_processed_queries(dataset_name)
    qrels = load_qrels(dataset_name)

    if dataset_name == "scifact":
        train_ids, val_ids = split_query_ids(qrels["train"].keys(), SCIFACT_VALIDATION_RATIO, SEED)
        original_train = qrels["train"]

        qrels["train"] = select_qrels(original_train, train_ids)
        qrels["validation"] = select_qrels(original_train, val_ids)

    retriever_ids, router_ids = split_query_ids(qrels["train"].keys(), ROUTER_TRAIN_RATIO, SEED + 1)

    qrels["retriever_train"] = select_qrels(qrels["train"], retriever_ids)
    qrels["router_train"] = select_qrels(qrels["train"], router_ids)

    partitions = [
        "retriever_train",
        "router_train",
        "validation",
        "test"
    ]

    all_ids = set()

    for split in partitions:
        current_ids = set(qrels[split])
        all_ids.update(current_ids)

    output_dir = Path("data/splits") / dataset_name

    for split, judgments in qrels.items():
        split_queries = {query_id: queries[query_id] for query_id in judgments}

        save_json(output_dir / f"queries_{split}.json", split_queries)
        save_json(output_dir / f"qrels_{split}.json", judgments)

        print(f"{split}: {len(split_queries)} queries")


def main():
    for dataset_name in DATASETS:
        print(f"\nPreparing {dataset_name.upper()}")
        prepare_dataset(dataset_name)


if __name__ == "__main__":
    main()
