# train.py
"""
Training script for Ticket Classification Pipeline
"""

import random
import argparse
from sklearn.model_selection import GridSearchCV

from .data_loader import load_ticket_dataset
from .pipeline import TicketClassifierPipeline
from .evaluate import evaluate_model
from .config import RANDOM_SEED

# ----------------------
# Hyperparameter grids 
# ----------------------
LOGREG_PARAM_GRID = {
    "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
    "clf__solver": ["liblinear"],  
}

SVC_PARAM_GRID = {
    "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
}

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

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run GridSearchCV for hyperparameter tuning",
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of CV folds (used only with --tune)",
    )

    args = parser.parse_args()

    print(f"Selected classifier: {args.classifier}")
    print(f"Tuning enabled: {args.tune}")

    # ----------------------
    # Load dataset
    # ----------------------
    X_train, y_train, X_val, y_val, X_test, y_test, TICKET_LABELS = load_ticket_dataset()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Top 5 labels: {TICKET_LABELS}")

    # ----------------------
    # Build the pipeline
    # ----------------------
    model = TicketClassifierPipeline(classifier=args.classifier)
    clf_pipeline = model.build_pipeline()

    # ----------------------
    # Train (with or without GridSearch)
    # ----------------------
    if args.tune:
        param_grid = LOGREG_PARAM_GRID if args.classifier == "logreg" else SVC_PARAM_GRID

        grid = GridSearchCV(
            estimator=clf_pipeline,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=args.cv,
            n_jobs=-1,
            verbose=2,
        )

        grid.fit(X_train, y_train)
        trained_model = grid.best_estimator_

        print("Best parameters:", grid.best_params_)
        print(f"Best CV score: {grid.best_score_:.4f}")
    else:
        clf_pipeline.fit(X_train, y_train)
        trained_model=clf_pipeline
        print(f"Pipeline trained on ticket dataset (Top 5 classes) using {args.classifier} (No Tuning).")

    # ----------------------
    # Evaluate
    # ----------------------
    evaluate_model(trained_model, X_val, y_val, show_confusion=False, labels=TICKET_LABELS)

if __name__ == "__main__":
    main()

