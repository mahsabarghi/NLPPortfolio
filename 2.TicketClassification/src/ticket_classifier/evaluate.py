# evaluate.py
"""
Evaluation script for Ticket Classification Pipeline
"""
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(pipeline, X, y, labels=None):
    """
    Evaluate the trained pipeline on given data

    Parameters:
    - pipeline: trained TicketClassifierPipeline object
    - X: list of texts
    - y: list of true labels
    - labels: list of all possible labels (ensures consistent evaluation reports)
    """
    if not X or not y:
        print("No evaluation data provided.")
        return None
    
    # predict labels for given data
    y_pred = pipeline.predict(X)

    # accuracy
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # classification report
    report = classification_report(y, y_pred, labels=labels)
    print("Classification report:\n")
    print(report)

    return acc, report