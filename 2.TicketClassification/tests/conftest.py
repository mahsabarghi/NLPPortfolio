import pytest
from ticket_classifier.data_loader import load_ticket_dataset
from ticket_classifier.pipeline import TicketClassifierPipeline

@pytest.fixture
@pytest.mark.slow
def dataset():
    """Load top 5 class ticket dataset."""
    return load_ticket_dataset()

@pytest.fixture
def dummy_dataset():
    X_dummy = [
        "My credit card was charged incorrectly",
        "Cannot access checking account",
        "Mortgage payment issue",
        "Credit report has an error",
        "Need help with prepaid card"
    ]
    y_dummy = [
        "Credit card",
        "Checking or savings account",
        "Mortgage",
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Credit card or prepaid card"
    ]
    return X_dummy, y_dummy

@pytest.fixture
def pipeline_logreg():
    """Return a pipeline with LogisticRegression."""
    return TicketClassifierPipeline(classifier='logreg')

@pytest.fixture
def pipeline_svc():
    """Return a pipeline with LinearSVC."""
    return TicketClassifierPipeline(classifier='svc')




