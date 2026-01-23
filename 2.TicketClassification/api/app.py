# app.py
"""
FastAPI app for Ticket Classification API
Dummy pipeline for initial testing
"""

from fastapi import FastAPI
from joblib import dump, load
from src.ticket_classifier.pipeline import TicketClassifierPipeline
from api.schemas import Ticket, Prediction
import numpy as np

# ----------------------
# Initialize FastAPI
# ----------------------
app = FastAPI(
    title="Ticket Classification API",
    description="Predict ticket categories using a scikit-learn pipeline",
    version="0.1"
)

# ----------------------
# Prepare dummy pipeline
# ----------------------
pipeline = TicketClassifierPipeline()
pipeline.build_pipeline()

# Minimal dummy training data
X_dummy = [
    "I need a refund",
    "App is crashing",
    "I want to change my subscription"
]
y_dummy = [
    "billing",
    "technical_issue",
    "feature_request"
]

pipeline.fit(X_dummy, y_dummy)

# Save dummy pipeline
dump(pipeline, "models/dummy_pipeline.joblib")

# ----------------------
# Predict endpoint
# ----------------------
@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket):
    # Predict label
    prediction = pipeline.predict([ticket.text])[0]

    # Predict confidence (probability)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([ticket.text])[0]
        # Take max probability as confidence
        confidence = float(np.max(probs))
    else:
        # Dummy pipelines may not support predict_proba
        confidence = 1.0  # set full confidence for dummy

    return Prediction(
        text=ticket.text,
        predicted_label=prediction,
        confidence=confidence
    )
