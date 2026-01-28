import pytest

# Hugging Face Dataset abstraction used for NLP pipelines
from datasets import Dataset

# Tokenizer for pretrained transformer models
from transformers import AutoTokenizer

# Centralized config for reproducibility 
from ticket_classifier.config import (
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_MAX_LEN,
)

@pytest.mark.slow
def test_transformer_tokenization_smoke():
    """
    Smoke test for transformer data preparation.

    This test verifies that:
    - raw text + labels can be wrapped into a Hugging Face Dataset
    - the pretrained tokenizer runs without errors
    - tokenization outputs the expected fields
    - tokenized sequences respect max length constraints

    The test intentionally:
    - uses a tiny in-memory dataset
    - does NOT train a model
    - avoids dependency on the real ticket dataset
    """

    # ---------------------------------------------------------
    # 1) Create a tiny synthetic dataset
    #    This keeps the test fast and deterministic
    # ---------------------------------------------------------
    texts = [
        "My bank transfer went to the wrong person.",
        "My credit card was charged twice.",
        "Mortgage payment was misapplied.",
    ]

    labels = ["A", "B", "C"]

    # Map string labels to numeric IDs (as required by transformers)
    label2id = {name: i for i, name in enumerate(sorted(set(labels)))}

    # ---------------------------------------------------------
    # 2) Build a Hugging Face Dataset
    # ---------------------------------------------------------
    ds = Dataset.from_dict(
        {
            "text": texts,
            "label": [label2id[y] for y in labels],
        }
    )

    # ---------------------------------------------------------
    # 3) Load pretrained tokenizer
    #    This defines how raw text is converted into token IDs
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    # ---------------------------------------------------------
    # 4) Define tokenization function
    #    - truncates long sequences
    #    - applies model-specific tokenization rules
    # ---------------------------------------------------------
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=TRANSFORMER_MAX_LEN,
        )

    # ---------------------------------------------------------
    # 5) Apply tokenization to the dataset
    #    - remove raw text after tokenization
    # ---------------------------------------------------------
    tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    # ---------------------------------------------------------
    # 6) Assertions: verify expected transformer inputs
    # ---------------------------------------------------------
    assert "input_ids" in tok.column_names
    assert "attention_mask" in tok.column_names
    assert "label" in tok.column_names

    # ---------------------------------------------------------
    # 7) Validate a single tokenized sample
    # ---------------------------------------------------------
    row0 = tok[0]

    # Labels must be integers
    assert isinstance(row0["label"], int)

    # Sequence length constraints must be respected
    assert len(row0["input_ids"]) <= TRANSFORMER_MAX_LEN

    # input_ids and attention_mask must align
    assert len(row0["input_ids"]) == len(row0["attention_mask"])
