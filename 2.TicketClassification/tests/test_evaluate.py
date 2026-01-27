# tests/test_evaluate.py
import pytest
from ticket_classifier.evaluate import evaluate_model
from ticket_classifier.pipeline import TicketClassifierPipeline

@pytest.mark.slow
def test_evaluate_model(dummy_dataset):
    X, y = dummy_dataset
    pipeline = TicketClassifierPipeline(classifier='logreg')
    pipeline.fit(X, y)
    
    metrics = evaluate_model(pipeline, X, y, show_confusion=False)
    
    # evaluate_model should return (acc, report)
    # evaluate_model should return (accuracy, report)
    assert isinstance(metrics, tuple)
    assert len(metrics) == 2

    acc, report = metrics

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report
