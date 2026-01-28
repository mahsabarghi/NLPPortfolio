# train_transformer.py

from __future__ import annotations

# Hugging Face Dataset: lightweight wrapper around tabular / text data
# that integrates cleanly with tokenizers, PyTorch, and Trainer
from datasets import Dataset

# Hugging Face tokenizer for pretrained transformer models + model for sequence classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

import torch

# Project-specific data loader (keeps same splits as classical models)
from ticket_classifier.data_loader import load_ticket_dataset

# Centralized config for reproducibility and tuning
from ticket_classifier.config import TRANSFORMER_MODEL_NAME, TRANSFORMER_MAX_LEN

def main() -> None:
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
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    # ---------------------------------------------------------
    # 5) Tokenization function
    #    - truncates long texts
    #    - applies model-specific tokenization rules
    # ---------------------------------------------------------
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=TRANSFORMER_MAX_LEN)

    # ---------------------------------------------------------
    # 6) Apply tokenizer to entire datasets
    #    - remove raw text column after tokenization
    #    - keep only numerical features needed by the model
    # ---------------------------------------------------------
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    # ---------------------------------------------------------
    # 7) Set PyTorch tensor format
    #    This makes the dataset compatible with DataLoader / Trainer
    # ---------------------------------------------------------
    cols = ["input_ids", "attention_mask", "label"]
    train_tok.set_format(type="torch", columns=cols)
    val_tok.set_format(type="torch", columns=cols)

    # ---------------------------------------------------------
    # 8) Load pretrained transformer model for classification
    # ---------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    # # ---------------------------------------------------------
    # # 8) Sanity checks
    # # ---------------------------------------------------------
    # print("\nTokenized datasets:")
    # print("train:", train_tok)
    # print("val:  ", val_tok)

    # print("\nOne sample (train[0]):")

    # sample = train_tok[0]
    # print("\nSample item structure:")
    # print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in sample.items()})
    # print("label id:", int(sample["label"]), "=>", id2label[int(sample["label"])])

    # print("\nDone.")

    # ---------------------------------------------------------
    # 9) Single forward pass sanity check (no training)
    #    Use a data collator to pad variable-length sequences in the batch
    # ---------------------------------------------------------
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


if __name__ == "__main__":
    main()
