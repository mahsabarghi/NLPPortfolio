# pipeline.py

# from sklearn.dummy import DummyClassifier # Just for a dummy run
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from .config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE
)
class TicketClassifierPipeline:
    def __init__(self, classifier='logreg'):
        """
        classifier:'logreg' for LogisticRegression, 'svc' for LinearSVC
        """
        self.classifier_name = classifier
        self.pipeline = None

    def build_pipeline(self):
        """
        Build the sklearn pipeline with vectorizer, and classifier
        """
        if self.classifier_name =='logreg':
             clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        elif self.classifier_name == 'svc':
            clf = LinearSVC(max_iter=5000, C=1.5, class_weight='balanced', loss='squared_hinge')
        else:
            raise ValueError(f"Classifier {self.classifier_name} is not supported.")


        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, stop_words='english')),
            ('clf', clf) 
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

