# app.py
"""
FastAPI app for Ticket Classification API
Serves a fine-tuned Hugging Face transformer model (RoBERTa)
"""

import torch
from pathlib import Path

from fastapi import FastAPI
from api.schemas import Ticket, Prediction
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------
# Config
# ----------------------
MODEL_DIR = Path("artifacts/transformer_runs/roberta-base_len128_n11711_e3.0/best_model")
MAX_LEN = 128

# ----------------------
# Initialize FastAPI
# ----------------------
app = FastAPI(
    title="Ticket Classification API",
    description="Predict ticket categories using a fine-tuned transformer model",
    version="1.0"
)

# ----------------------
# Load model once at startup (fast inference)
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Use CPU by default (portable & consistent with training runs)
DEVICE = torch.device("cpu")
model.to(DEVICE)
# ----------------------
# Predict endpoint
# ----------------------
@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket):
    # Tokenize input text
    inputs = tokenizer(
        ticket.text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    # Map id -> label (saved inside the model config)
    pred_label = model.config.id2label.get(pred_id, str(pred_id))

    return Prediction(
        text=ticket.text,
        predicted_label=pred_label,
        confidence=confidence
    )

@app.get("/health")
def health():
    return {"status": "ok"}
