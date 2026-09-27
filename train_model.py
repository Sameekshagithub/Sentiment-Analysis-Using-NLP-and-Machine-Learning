"""
train_model.py

Loads the synthetic review dataset, cleans and vectorizes the text with
TF-IDF, trains a Logistic Regression classifier, evaluates it on a held
out test split, and saves the trained model + vectorizer to disk so the
Flask app can load them without retraining.

Run:
    python train_model.py
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "evaluation_report.txt")


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run data/generate_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review", "sentiment"])
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} rows. Class counts:\n{df['sentiment'].value_counts()}")

    print("Cleaning and preprocessing text...")
    df["clean_review"] = df["review"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train_vec, y_train)

    print("Evaluating on held out test data...")
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print("Classification report:")
    print(report)
    print("Confusion matrix (rows = actual, columns = predicted)")
    print("Labels order:", list(model.classes_))
    print(matrix)

    with open(REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix (rows = actual, columns = predicted)\n")
        f.write(f"Labels order: {list(model.classes_)}\n")
        f.write(str(matrix))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
