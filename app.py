"""
app.py

Flask web app for the sentiment analysis project. Loads the trained
TF-IDF vectorizer and Logistic Regression model, serves a small single
page frontend, and exposes a /predict endpoint the frontend calls with
whatever text the user typed in.
"""

import os
import pickle

from flask import Flask, render_template, request, jsonify

from preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

app = Flask(__name__)

_model = None
_vectorizer = None


def load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                "Model files not found. Run train_model.py before starting the app."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
    return _model, _vectorizer


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    model, vectorizer = load_artifacts()

    cleaned = preprocess(text)
    if not cleaned:
        return jsonify({"error": "Could not extract any meaningful words from that text."}), 400

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    class_confidences = {
        label: round(float(prob) * 100, 2)
        for label, prob in zip(model.classes_, probabilities)
    }
    confidence = class_confidences[prediction]

    return jsonify({
        "sentiment": prediction,
        "confidence": confidence,
        "breakdown": class_confidences,
        "cleaned_text": cleaned,
    })


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True, port=5000)
