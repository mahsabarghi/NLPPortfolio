# tests/test_pipeline.py
import pytest
from ticket_classifier.pipeline import TicketClassifierPipeline
from ticket_classifier.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE


def test_pipeline_build_logreg():
    """Test building LogisticRegression pipeline and check steps"""
    pipeline_logreg = TicketClassifierPipeline(classifier='logreg')
    pipeline = pipeline_logreg.build_pipeline()
    
    # Pipeline should exist
    assert pipeline is not None
    
    # Check steps
    steps = dict(pipeline.named_steps)
    assert 'tfidf' in steps
    assert 'clf' in steps
    assert steps['clf'].__class__.__name__ == 'LogisticRegression'

    # Check TF-IDF parameters
    tfidf = steps['tfidf']
    assert tfidf.max_features == TFIDF_MAX_FEATURES
    assert tfidf.ngram_range == TFIDF_NGRAM_RANGE
    assert tfidf.stop_words == "english"


def test_pipeline_build_svc():
    """Test building LinearSVC pipeline and check steps"""
    pipeline_svc = TicketClassifierPipeline(classifier='svc')
    pipeline = pipeline_svc.build_pipeline()
    
    # Pipeline should exist
    assert pipeline is not None
    
    # Check steps
    steps = dict(pipeline.named_steps)
    assert 'tfidf' in steps
    assert 'clf' in steps
    assert steps['clf'].__class__.__name__ == 'LinearSVC'

    # Check TF-IDF parameters
    tfidf = steps['tfidf']
    assert tfidf.max_features == TFIDF_MAX_FEATURES
    assert tfidf.ngram_range == TFIDF_NGRAM_RANGE
    assert tfidf.stop_words == "english"


def test_pipeline_fit_predict(dummy_dataset):
    """Test fitting and predicting with the pipeline"""
    X, y = dummy_dataset  # dummy_dataset from conftest.py
    pipeline = TicketClassifierPipeline(classifier='logreg')
    pipeline.fit(X, y)
    
    # Predict
    y_pred = pipeline.predict(X)
    
    # Check output
    assert len(y_pred) == len(y)
    assert all(isinstance(label, str) for label in y_pred)

# Integration tests
@pytest.mark.slow
def test_pipeline_integration_logreg(dataset):
    """Integration test using real top 5 dataset for LogisticRegression"""
    X_train, y_train, X_val, y_val, X_test, y_test, TICKET_LABELS = dataset
    pipeline = TicketClassifierPipeline(classifier='logreg')
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    
    assert len(y_pred) == len(y_val)
    assert set(y_pred).issubset(set(TICKET_LABELS))
    assert len(set(y_pred)) <= len(TICKET_LABELS)

@pytest.mark.slow
def test_pipeline_integration_svc(dataset):
    """Integration test using real top 5 dataset"""
    X_train, y_train, X_val, y_val, X_test, y_test, TICKET_LABELS = dataset
    pipeline = TicketClassifierPipeline(classifier='svc')
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    
    # Check output
    assert len(y_pred) == len(y_val)
    assert set(y_pred).issubset(set(TICKET_LABELS))
    assert len(set(y_pred)) <= len(TICKET_LABELS)

