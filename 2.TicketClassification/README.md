# Ticket Classification with NLP

An end-to-end NLP project for automatic ticket classification using classical machine learning techniques.
The project is designed with production-style structure, testing, and extensibility in mind, and serves as a strong baseline before moving to transformer-based models.

## 📌 Project Overview

Customer support teams receive large volumes of free-text tickets every day.
This project focuses on classifying support tickets into predefined categories based on the textual description provided by users.

The current implementation establishes a strong classical NLP baseline using TF-IDF and linear classifiers, with proper evaluation, testing, and hyperparameter tuning.

## 📊 Dataset

**Source:** Financial complaint/ticket dataset

**Input:** Free-text description (complaint_what_happened)

**Target:** Product category (product)

**Preprocessing:**

- Remove missing text or labels

- Keep top 5 most frequent classes

- Divide data into train/validation/test subsets

## 🧠 Modeling Approach
  **🔹Text Representation**

    - TF-IDF Vectorization

        - Configurable via config.py

        - N-grams: (1, 2)

        - Max_features: 5000

        - Stop-word removal (english)

**🔹 Classifiers**

  - Logistic Regression

    - class_weight="balanced"

  - Linear SVM (LinearSVC)

    - class_weight="balanced"

Both models are implemented through a single configurable pipeline.

## 🔧 Training & Hyperparameter Tuning

Training is done via a command-line interface (CLI).
1. Training without tuning:
   
   PYTHONPATH=src python -m ticket_classifier.train --classifier logreg
   
   PYTHONPATH=src python -m ticket_classifier.train --classifier svc

3. Training with GridSearchCV:
   
   PYTHONPATH=src python -m ticket_classifier.train --classifier logreg --tune
   
   PYTHONPATH=src python -m ticket_classifier.train --classifier svc --tune

## 📈 Evaluation

Evaluation includes:

- Accuracy

- Precision / Recall / F1-score (per class)

- Macro & weighted averages

**Example performance (Top 5 classes):**

Accuracy: ~78%

Weighted F1: ~0.78

> These results are typical for TF-IDF-based models on semantically overlapping ticket categories.

## Testing

The project includes a pytest-based test covering:

- Pipeline construction

- TF-IDF configuration consistency

- Fit / predict sanity checks

- Data loading validation

- Integration tests on real data
  
> For running the tests: PYTHONPATH=src pytest

## ⚙️ Configuration

All reusable parameters are centralized in config.py, including:

- TF-IDF settings

- Random seed

- Dataset split sizes

- Paths

> This ensures consistency between training, testing, and experiments.

## 🚀 Next Steps

Planned improvements:

1. Transformer-based embeddings (BERT / Sentence Transformers)

2. Fine-tuned transformer classifiers

3. Model persistence & inference API

4. CI pipeline (GitHub Actions)

5. Error analysis & label refinement
  
## Status
🚧 Project setup in progress

