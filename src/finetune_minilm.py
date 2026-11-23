import argparse

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

from utils import DATASETS, PROJECT_ROOT

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = PROJECT_ROOT / "models" / "minilm_finetuned"


def load_training_pairs():
    filenames = [PROJECT_ROOT / "data" / "training" / dataset / "retriever_model_inputs.jsonl" for dataset in DATASETS]
    train_dataset = load_dataset("json", data_files=[str(path) for path in filenames], split="train")

    return train_dataset


def fine_tune(train_dataset):
    model = SentenceTransformer(BASE_MODEL)
    print("Tokenizer:", type(model.tokenizer).__name__)
    print("Max sequence length:", model.max_seq_length)
    print("Embedding dimensions:", model.get_sentence_embedding_dimension())

    loss = losses.MultipleNegativesRankingLoss(model)

    # TODO: We should change these when training
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=6,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    final_dir = OUTPUT_DIR / "final"
    model.save(str(final_dir))
    print("Fine-tuned model saved to:", final_dir)


def main():
    train_dataset = load_training_pairs()
    print("Training pairs:", len(train_dataset))
    print("Columns:", train_dataset.column_names)
    print("Sample anchor:", train_dataset[0]["anchor"][:120])
    print("Sample positive:", train_dataset[0]["positive"][:120])

    # fine_tune(train_dataset)


if __name__ == "__main__":
    main()
