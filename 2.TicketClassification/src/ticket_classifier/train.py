# train.py
"""
Training script for Ticket Classification Pipeline
"""

from pathlib import Path
import json
import random

from .pipeline import TicketClassifierPipeline
from .config import RAW_DATA_DIR, SPLITS_DIR, RANDOM_SEED

# ----------------------
# Set seed for reproducibility
# ----------------------
random.seed(RANDOM_SEED)

# ----------------------
# Load dataset (placeholder)
# ----------------------
def load_dataset():
    """
    Load train/validation data from raw JSON files
    Currently, this is a placeholder that returns empty lists
    """
    train_file = RAW_DATA_DIR / "train.json"
    val_file = RAW_DATA_DIR / "val.json"

    # TODO: replace with real file loading
    X_train, y_train = [], []
    X_val, y_val = [], []

    return X_train, y_train, X_val, y_val

# ----------------------
# Main training function
# ----------------------
def main():
    X_train, y_train, X_val, y_val = load_dataset()

    # Initialize the pipeline
    clf_pipeline = TicketClassifierPipeline()
    clf_pipeline.build_pipeline()

    # Fit the pipeline (placeholder, no real data yet)
    if X_train and y_train:
        clf_pipeline.fit(X_train, y_train)
        print("Pipeline trained successfully")
    else:
        print("No training data found. Pipeline not trained.")

if __name__ == "__main__":
    main()

