# train_transformer.py

from __future__ import annotations

# Hugging Face Dataset: lightweight wrapper around tabular / text data
# that integrates cleanly with tokenizers, PyTorch, and Trainer
from datasets import Dataset

# Hugging Face components:
# - AutoTokenizer: converts raw text into model-specific token IDs
# - AutoModelForSequenceClassification: pretrained transformer with a classification head
# - DataCollatorWithPadding: dynamically pads variable-length sequences for batching
# - TrainingArguments: configuration object for the training loop (epochs, batch size, logging, etc.)
# - Trainer: high-level API that handles training, evaluation, and checkpointing
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)

import torch
import argparse

# Project-specific data loader (keeps same splits as classical models)
from ticket_classifier.data_loader import load_ticket_dataset

# Centralized config for reproducibility and tuning
from ticket_classifier.config import RANDOM_SEED, TRANSFORMER_MODEL_NAME, TRANSFORMER_MAX_LEN

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics from Trainer predictions.
    eval_pred is a tuple: (logits, labels)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def main() -> None:
    # ---------------------------------------------------------
    # CLI args
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train transformer for ticket classification")
    parser.add_argument("--sanity-forward", action="store_true", help="Run a single forward-pass sanity check and exit")
    parser.add_argument("--train-n", type=int, default=3000, help="Number of training samples to use")
    parser.add_argument("--val-n", type=int, default=1000, help="Number of validation samples to use")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train-batch", type=int, default=4, help="Train batch size (CPU-friendly)")
    parser.add_argument("--eval-batch", type=int, default=16, help="Eval batch size")
    parser.add_argument("--run-name", type=str, default="distilbert_run", help="Name for output folder")
    parser.add_argument("--model-name", type=str, default=TRANSFORMER_MODEL_NAME, help="Hugging Face model checkpoint (e.g., distilbert-base-uncased, roberta-base)")
    parser.add_argument("--max-len", type=int, default=TRANSFORMER_MAX_LEN, help="Max token length (e.g., 128, 256). Higher can improve accuracy but uses more memory.")
    args = parser.parse_args()

    if args.run_name == "distilbert_run":
        safe_model = args.model_name.replace("/", "_")
        args.run_name = f"{safe_model}_len{args.max_len}_n{args.train_n}_e{args.epochs}"

    
    # ---------------------------------------------------------
    # 1) Load pre-split dataset (train / val / test)
    #    This reuses the exact same data pipeline as TF-IDF models
    # ---------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_ticket_dataset()

    # ---------------------------------------------------------
    # 2) Create label ↔ id mappings
    #    Transformers expect numeric labels, not strings
    # ---------------------------------------------------------
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    # ---------------------------------------------------------
    # 3) Wrap raw text + labels into Hugging Face Dataset objects
    #    This is the standard data container for HF pipelines
    # ---------------------------------------------------------
    train_ds = Dataset.from_dict(
        {"text": X_train, "label": [label2id[y] for y in y_train]}
    )
    val_ds = Dataset.from_dict(
        {"text": X_val, "label": [label2id[y] for y in y_val]}
    )

    # ---------------------------------------------------------
    # 4) Load pretrained tokenizer
    #    This defines how raw text is converted into token IDs
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---------------------------------------------------------
    # 5) Tokenization function
    #    - truncates long texts
    #    - applies model-specific tokenization rules
    # ---------------------------------------------------------
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=args.max_len)

    # ---------------------------------------------------------
    # 6) Apply tokenizer to entire datasets
    #    - remove raw text column after tokenization
    #    - keep only numerical features needed by the model
    # ---------------------------------------------------------
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    # ---------------------------------------------------------
    # 7) Load pretrained transformer model for classification
    # ---------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    # ---------------------------------------------------------
    # Optional: Single forward pass sanity check (no training)
    # ---------------------------------------------------------
    if args.sanity_forward:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

        batch_items = [train_tok[i] for i in range(4)]  # list of dicts
        batch = data_collator(batch_items)              # pads + stacks tensors

        # labels are not included by the collator automatically from "label" sometimes, so add them explicitly
        batch["labels"] = torch.stack([x["label"] for x in batch_items])

        with torch.no_grad():
            outputs = model(**batch) # Pass the whole batch

        # Logits shape should be: (batch_size, num_labels)
        print("\nForward pass sanity check:")
        print("loss:", float(outputs.loss))
        print("logits shape:", tuple(outputs.logits.shape))

        preds = torch.argmax(outputs.logits, dim=-1)
        print("pred ids:", preds.tolist())
        print("true ids:", batch["labels"].tolist())
        print("pred labels:", [id2label[int(i)] for i in preds.tolist()])

        print("\nDone.")
        return # to exit the current function (which is main) to do only sanity check
    # ---------------------------------------------------------
    # 8) Training subset (configurable via CLI)
    # ---------------------------------------------------------
    train_n = min(args.train_n, len(train_tok))
    val_n = min(args.val_n, len(val_tok))

    train_small = train_tok.select(range(train_n))
    val_small = val_tok.select(range(val_n))

    test_ds = Dataset.from_dict({"text": X_test, "label": [label2id[y] for y in y_test]})
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    # ---------------------------------------------------------
    # 9) Data collator for dynamic padding (correct batching)
    # ---------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---------------------------------------------------------
    # 10) Trainer configuration 
    # ---------------------------------------------------------
    args_train = TrainingArguments(
        output_dir=f"artifacts/transformer_runs/{args.run_name}",
        eval_strategy="epoch",
        save_strategy="epoch",          # save each epoch so we can keep best later
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        seed=RANDOM_SEED,
        report_to="none",
        use_cpu=True,
        use_mps_device=False,  # <- force CPU on Mac

        # --- Best model selection ---
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        # Keep disk usage reasonable
        save_total_limit=2,
    )


    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_small,
        eval_dataset=val_small,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---------------------------------------------------------
    # 11) Train + evaluate
    # ---------------------------------------------------------
    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_small)
    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    test_metrics = trainer.evaluate(eval_dataset=test_tok)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ---------------------------------------------------------
    # 12) Save model + tokenizer (to do inference next)
    # ---------------------------------------------------------
    save_dir = f"artifacts/transformer_runs/{args.run_name}/best_model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\nSaved best model to:", save_dir)
    print("Done.")

if __name__ == "__main__":
    main()
