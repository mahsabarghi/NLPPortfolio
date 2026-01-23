# train.py
"""
Training script for Ticket Classification Pipeline
"""

import random
from .data_loader import load_ticket_dataset
from .pipeline import TicketClassifierPipeline
from .evaluate import evaluate_model
from .config import RANDOM_SEED

# ----------------------
# Set seed for reproducibility
# ----------------------
random.seed(RANDOM_SEED)

# ----------------------
# Main training function
# ----------------------
def main():
    # ----------------------
    # Load dataset
    # ----------------------
    X_train, y_train, X_val, y_val, X_test, y_test, TICKET_LABELS = load_ticket_dataset()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Top 5 labels: {TICKET_LABELS}")

    # ----------------------
    # Initialize the pipeline
    # ----------------------
    clf_pipeline = TicketClassifierPipeline()
    clf_pipeline.build_pipeline()

    # ----------------------
    # Fit the pipeline
    # ----------------------
    clf_pipeline.fit(X_train, y_train)
    print("Pipeline trained on ticket dataset (Top 5 classes).")

    # ----------------------
    # Evaluate
    # ----------------------
    evaluate_model(clf_pipeline, X_val, y_val, show_confusion=False, labels=TICKET_LABELS)

if __name__ == "__main__":
    main()

