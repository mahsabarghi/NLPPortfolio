# pipeline.py
"""
Skeleton for the Ticket Classification Pipeline
Preprocessing and model will live here
"""
from sklearn.dummy import DummyClassifier # Just for a dummy run
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class TicketClassifierPipeline:
    def __init__(self):
        # Placeholder for future pipeline
        self.pipeline = None

    def build_pipeline(self):
        """
        Build the sklearn pipeline with vectorizer, transformer, and classifier
        """
        # Example placeholder: we will fill this later
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),  # will use config ngram settings
            # ('tfidf', TfidfTransformer()),
            ('classifier', DummyClassifier(strategy='most_frequent'))  # For a dummy run
        ])
        return self.pipeline

    def fit(self, X, y):
        """
        Fit the pipeline on data
        """
        if self.pipeline is None:
            self.build_pipeline()
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict ticket categories
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built or fitted yet.")
        return self.pipeline.predict(X)
