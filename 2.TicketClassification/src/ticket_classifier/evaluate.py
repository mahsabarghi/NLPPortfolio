# evaluate.py
"""
Evaluation script for Ticket Classification Pipeline
"""
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(pipeline, X, y, show_confusion=True, labels=None):
    """
    Evaluate the trained pipeline on given data

    Parameters:
    - pipeline: trained TicketClassifierPipeline object
    - X: list of texts
    - y: list of true labels
    - labels: list of all possible labels (ensures consistent evaluation reports)
    - show_confusion: whether to show confusion matrix
    """
    if not X or not y:
        print("No evaluation data provided.")
        return None
    
    # predict labels for given data
    y_pred = pipeline.predict(X)

    # accuracy
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")

     # infer labels if not provided
    if labels is None:
        labels = sorted(list(set(y)))

    # classification report
    report = classification_report(y, y_pred, labels=labels, zero_division=0, digits=4)
    print("Classification report:\n")
    print(report)

    # confusion matrix
    if show_confusion:
        cm = confusion_matrix(y, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    return acc, report