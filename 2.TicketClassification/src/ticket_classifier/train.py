# train.py
"""import argparse
Training script for Ticket Classification Pipeline
"""

import random
import argparse
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
    # Parse arguments
    # ----------------------
    parser = argparse.ArgumentParser(description="Train Ticket Classification Pipeline")
    parser.add_argument(
        "--classifier",
        type=str,
        default="logreg",
        choices=["logreg", "svc"],
        help="Classifier to train: logreg (LogisticRegression) or svc (LinearSVC)"
    )
    args = parser.parse_args()

    print(f"Selected classifier: {args.classifier}")

    # ----------------------
    # Load dataset
    # ----------------------
    X_train, y_train, X_val, y_val, X_test, y_test, TICKET_LABELS = load_ticket_dataset()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Top 5 labels: {TICKET_LABELS}")

    # ----------------------
    # Initialize the pipeline
    # ----------------------
    clf_pipeline = TicketClassifierPipeline(classifier=args.classifier)
    clf_pipeline.build_pipeline()

    # ----------------------
    # Fit the pipeline
    # ----------------------
    clf_pipeline.fit(X_train, y_train)
    print(f"Pipeline trained on ticket dataset (Top 5 classes) using {args.classifier}.")

    # ----------------------
    # Evaluate
    # ----------------------
    evaluate_model(clf_pipeline, X_val, y_val, show_confusion=False, labels=TICKET_LABELS)

if __name__ == "__main__":
    main()

