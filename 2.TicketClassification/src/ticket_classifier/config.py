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
#  Parameters
# ----------------------
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# TF-IDF configuration
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)


