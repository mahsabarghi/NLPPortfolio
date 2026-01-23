# schemas.py
"""
Pydantic models for Ticket Classification API
"""

from pydantic import BaseModel

# Request schema
class Ticket(BaseModel):
    text: str  # the ticket text to classify

# Response schema

class Prediction(BaseModel):
    text: str                # original ticket text
    predicted_label: str     # predicted category
    confidence: float        # probability/confidence score
