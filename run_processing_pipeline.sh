
#!/usr/bin/env bash
set -euo pipefail

python src/parse_datasets.py
python src/preprocess_text.py
python src/prepare_splits.py
python src/prepare_training_pairs.py
python src/prepare_router_features.py
python src/build_bm25_index.py

echo "=== All preprocessing stages completed successfully! ==="
