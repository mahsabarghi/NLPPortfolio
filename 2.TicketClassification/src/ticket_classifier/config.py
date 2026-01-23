# config.py
# Configuration for Ticket Classification project

from pathlib import Path

# ----------------------
# Paths
# ----------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to 2.TicketClassification
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODEL_DIR = BASE_DIR / "models"

# ----------------------
# Seed for reproducibility
# ----------------------
RANDOM_SEED = 42

# ----------------------
# Ticket categories (labels)
# ----------------------
TICKET_LABELS = [
    "billing",
    "technical_issue",
    "account_management",
    "feature_request",
    "general_inquiry"
]

# ----------------------
# Hyperparameters
# ----------------------
MAX_FEATURES = 5000  # max vocabulary size for text vectorizer
NGRAM_RANGE = (1, 2)  # unigrams + bigrams
TEST_SIZE = 0.2
VAL_SIZE = 0.1

