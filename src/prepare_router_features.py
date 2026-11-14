
import json
import re
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


DATASETS = ["fiqa", "scifact"]

# TODO: When doing the router training we'll have to check these
N_COMPONENTS = 128
SEED = 6
MIN_DF = 2
MAX_FEATURES = 20000

def load_split_queries(split_name):
    rows = []

    for dataset_name in DATASETS:
        path = (Path("data/splits") / dataset_name / f"queries_{split_name}.json")

        with path.open("r", encoding="utf-8") as file:
            queries = json.load(file)

        for query_id, query_text in sorted(queries.items()):
            rows.append({
                "dataset": dataset_name,
                "query_id": query_id,
                "text": query_text
            })

    return rows


def extract_query_statistics(rows):
    features = []

    for row in rows:
        text = row["text"]

        tokens = re.findall(r"\b\w+\b", text.lower())
        char_count = len(text)
        token_count = len(tokens)

        unique_ratio = (len(set(tokens)) / token_count if token_count > 0 else 0.0)
        avg_token_length = (sum(len(token) for token in tokens) / token_count if token_count > 0 else 0.0)

        features.append([char_count, token_count, unique_ratio, avg_token_length])

    return np.asarray(features, dtype=np.float64)



def fit_feature_extractor(train_rows):
    texts = [row["text"] for row in train_rows]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True
    )

    train_tfidf = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)

    train_svd = svd.fit_transform(train_tfidf)
    train_stats = extract_query_statistics(train_rows)

    train_features = np.hstack([train_svd, train_stats])

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    return vectorizer, svd, scaler, train_features


def transform_queries(rows, vectorizer, svd, scaler):
    texts = [row["text"] for row in rows]

    tfidf_features = vectorizer.transform(texts)
    svd_features = svd.transform(tfidf_features)

    stats = extract_query_statistics(rows)

    features = np.hstack([svd_features, stats])

    return scaler.transform(features)



def save_features(path, rows, features):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row, vector in zip(rows, features):
            record = {
                "dataset": row["dataset"],
                "query_id": row["query_id"],
                "features": vector.tolist()
            }

            file.write(json.dumps(record, ensure_ascii=False) + "\n")



def main():
    output_dir = Path("data/router_features")
    model_dir = Path("models")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split_queries("router_train")

    print(f"Router training queries: {len(train_rows)}")

    vectorizer, svd, scaler, train_features = fit_feature_extractor(train_rows)

    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Router feature dimensions: {train_features.shape[1]}")
    print("SVD explained variance:", round(float(svd.explained_variance_ratio_.sum()), 4))

    save_features(output_dir / "router_train.jsonl", train_rows, train_features)

    for split in ["validation", "test"]:
        rows = load_split_queries(split)

        features = transform_queries(rows, vectorizer, svd, scaler)
        save_features(output_dir / f"{split}.jsonl", rows, features)

        print(f"{split}: {len(rows)} queries processed")

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "svd": svd,
            "scaler": scaler
        }, model_dir / "router_feature_extractor.joblib"
    )

    feature_names = [f"svd_{i:03d}" for i in range(N_COMPONENTS)] + [
        "char_count",
        "token_count",
        "unique_token_ratio",
        "avg_token_length"
    ]

    metadata = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "fit_split": "router_train",
        "datasets": DATASETS,
        "svd_explained_variance_ratio": float(svd.explained_variance_ratio_.sum())
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nRouter feature extraction completed!")


if __name__ == "__main__":
    main()
