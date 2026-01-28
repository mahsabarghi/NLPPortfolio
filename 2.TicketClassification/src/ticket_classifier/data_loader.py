# data_loader.py
"""
Load ticket classification dataset
"""

from pathlib import Path
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from .config import RAW_DATA_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE

def load_ticket_dataset(test_size=TEST_SIZE, val_size=VAL_SIZE, random_seed=RANDOM_SEED):
    """
    Load ticket dataset, keep top 5 classes,  and split into train, validation, and test sets.

    Parameters:
    - test_size: fraction of dataset for test
    - val_size: fraction of train+val used for validation
    - random_seed: for reproducibility

    Returns:
    - X_train, y_train, X_val, y_val, X_test, y_test, TICKET_LABELS
    """
    data_file = RAW_DATA_DIR / "tickets.json"
    
    # ----------------------
    # Load JSON
    # ----------------------
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # ----------------------
    # Filter out missing values
    # ----------------------
    filtered_data = [
        item for item in data
        if item.get("_source") 
        and item["_source"].get("complaint_what_happened")
        and item["_source"].get("product") # 'product' is the target label
    ]

    # ----------------------
    # Flatten texts and labels
    # ----------------------
    texts = [item["_source"]["complaint_what_happened"] for item in filtered_data]
    labels = [item["_source"]["product"] for item in filtered_data]  
    
    # ----------------------
    # Keep only top 5 classes by count
    # ----------------------
    counter = Counter(labels)
    top5_classes = [label for label, _ in counter.most_common(5)]

    texts_top5 = []
    labels_top5 = []
    for text, label in zip(texts, labels):
        if label in top5_classes:
            texts_top5.append(text)
            labels_top5.append(label)

    # ----------------------
    # Train/val/test split
    # ----------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts_top5, labels_top5, test_size=test_size, random_state=random_seed, stratify=labels_top5
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative_size, random_state=random_seed, stratify=y_trainval
    )

    # ----------------------
    # Dynamically get all labels
    # ----------------------
    TICKET_LABELS = sorted(list(set(y_train + y_val + y_test)))

    print(f"Total samples (Top 5 classes): {len(texts_top5)}, Total labels: {len(TICKET_LABELS)}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Top 5 labels: {TICKET_LABELS}")

    return X_train, y_train, X_val, y_val, X_test, y_test, TICKET_LABELS
