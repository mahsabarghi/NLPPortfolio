# train_transformer.py

from __future__ import annotations

from ticket_classifier.data_loader import load_ticket_dataset

def main() -> None:
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        labels,
    ) = load_ticket_dataset()

    print("\nSanity check:")
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    print(f"Labels ({len(labels)}): {labels}")

    print("\nSample text:")
    print(X_train[0][:500])
    print("Label:", y_train[0])

    print("\nDone.")


if __name__ == "__main__":
    main()
