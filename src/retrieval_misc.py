import json
import math
import time
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
TOP_K = 100


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split(dataset_name, split_name):
    split_directory = DATA_DIRECTORY / "splits" / dataset_name
    queries = load_json(split_directory / f"queries_{split_name}.json")
    qrels = load_json(split_directory / f"qrels_{split_name}.json")

    return queries, qrels


# ---------------------------
# Metrics
# ---------------------------


def ndcg_at_k(ranked_doc_ids, relevant, k=10):
    dcg = 0.0

    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        gain = relevant.get(doc_id, 0)
        dcg += gain / math.log2(rank + 1)

    ideal_gains = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    ideal_dcg = sum(
        gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1)
    )

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(ranked_doc_ids, relevant, k=100):
    relevant_ids = {doc_id for doc_id, gain in relevant.items() if gain > 0}

    if not relevant_ids:
        return 0.0

    nr_of_hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_ids)
    return nr_of_hits / len(relevant_ids)


def mrr_at_k(ranked_doc_ids, relevant, k=10):
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def compute_metrics(ranked_doc_ids, relevant):
    return {
        "ndcg@10": ndcg_at_k(ranked_doc_ids, relevant, 10),
        "recall@100": recall_at_k(ranked_doc_ids, relevant, 100),
        "mrr@10": mrr_at_k(ranked_doc_ids, relevant, 10),
    }


# ---------------------------
# Evaluation
# ---------------------------


class Retriever(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, k: int = TOP_K) -> tuple[list[str], list[float]]:
        """Returns (doc_id, scores) for the top-k documents, where the best is first."""


WARMUP_QUERIES = 5


def run_retriever(retriever: Retriever, queries, qrels, k=TOP_K, show_progress=True):
    query_ids = sorted(qrels.keys())

    for query_id in query_ids[:WARMUP_QUERIES]:
        retriever.search(queries[query_id], k)

    records = []

    for i, query_id in enumerate(query_ids, start=1):
        start = time.perf_counter()
        doc_ids, scores = retriever.search(queries[query_id], k)
        latency_ms = (time.perf_counter() - start) * 1000

        # Example structure:
        # {"query_id": "q42", "doc_ids": ["d9", "d3", ...], "scores": [12.31, 11.8, ...],
        # "ndcg@10": 0.67, "recall@100": 1.0, "mrr@10": 0.5, "latency_ms": 3.2}
        record = {
            "query_id": query_id,
            "doc_ids": doc_ids,
            "scores": [round(float(s), 6) for s in scores],
            **compute_metrics(doc_ids, qrels[query_id]),
            "latency_ms": latency_ms,
        }
        records.append(record)

        if show_progress and (i % 250 == 0 or i == len(query_ids)):
            print(f"{i}/{len(query_ids)} queries run", flush=True)

    return records


def summarize(records):
    latencies = np.array([r["latency_ms"] for r in records])

    return {
        "num_queries": len(records),
        "ndcg@10": float(np.mean([r["ndcg@10"] for r in records])),
        "recall@100": float(np.mean([r["recall@100"] for r in records])),
        "mrr@10": float(np.mean([r["mrr@10"] for r in records])),
        "latency_mean_ms": float(latencies.mean()),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def print_summary(name, summary):
    print(
        f"{name:<20} nDCG@10={summary['ndcg@10']:.4f}  "
        f"R@100={summary['recall@100']:.4f}  MRR@10={summary['mrr@10']:.4f}  "
        f"latency mean={summary['latency_mean_ms']:.2f}ms  "
        f"median={summary['latency_median_ms']:.2f}ms  p95={summary['latency_p95_ms']:.2f}ms"
    )
