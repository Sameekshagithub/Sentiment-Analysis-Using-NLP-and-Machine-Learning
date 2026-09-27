# Sentiment Analysis (NLP + Flask)

A small end-to-end sentiment analysis project: a synthetic review dataset,
a TF-IDF + Logistic Regression classifier, and a Flask web app with a
butter-yellow themed frontend where you can type in text and see the
predicted sentiment.

## Project layout

```
sentiment_analysis_project/
├── app.py                     # Flask app (routes + /predict API)
├── preprocess.py               # shared text cleaning / tokenizing / stemming
├── train_model.py               # trains and evaluates the model, saves it to disk
├── requirements.txt
├── data/
│   ├── generate_dataset.py     # builds the synthetic dataset
│   └── reviews_dataset.csv     # generated dataset (~1,266 labeled reviews)
├── model/
│   ├── sentiment_model.pkl     # trained Logistic Regression model (created by train_model.py)
│   ├── vectorizer.pkl          # fitted TF-IDF vectorizer (created by train_model.py)
│   └── evaluation_report.txt   # accuracy / precision / recall / confusion matrix
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Setup (VS Code / local machine)

1. Open this folder in VS Code.
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. (Optional) Regenerate the dataset. A dataset is already included, but
   you can rebuild it or tweak `data/generate_dataset.py` and rerun:
   ```
   python data/generate_dataset.py
   ```
5. Train the model. This reads `data/reviews_dataset.csv`, trains the
   classifier, prints an evaluation report, and saves the model +
   vectorizer into `model/`:
   ```
   python train_model.py
   ```
6. Run the web app:
   ```
   python app.py
   ```
7. Open `http://127.0.0.1:5000` in your browser, type in a sentence, and
   click "Analyze sentiment".

## About the dataset

The dataset is **synthetic** — it's generated from templates and word
lists in `data/generate_dataset.py`, combining subjects (product, phone,
hotel, service, etc.) with positive/negative/neutral adjectives and
sentence patterns, plus a smaller set of hand-written, less templated
examples for variety. There's no real customer data behind it. Because
of that, the reported accuracy (close to 99% on the held-out split) is
higher than you'd see on a real-world review dataset — the model is
picking up patterns in fairly consistent templates. If you want a more
realistic benchmark, swap in a public dataset (e.g. IMDB reviews,
Amazon reviews, or Sentiment140) with the same two columns (`review`,
`sentiment`) and rerun `train_model.py`.

## How the pipeline works

1. **Cleaning** (`preprocess.clean_text`): lowercases text, strips URLs,
   HTML tags, punctuation and digits.
2. **Tokenizing**: splits on whitespace.
3. **Stopword removal**: drops common low-signal words, but deliberately
   keeps negation words like "not" and "no" in the vocabulary, since
   "not good" and "good" mean opposite things.
4. **Stemming**: a small rule-based suffix stripper (playing/played/plays
   → play). It's lightweight on purpose so the project has no dependency
   on downloading external NLTK data.
5. **Feature extraction**: TF-IDF over unigrams and bigrams
   (`TfidfVectorizer(ngram_range=(1, 2), max_features=5000)`).
6. **Model**: `LogisticRegression` from scikit-learn, trained on an
   80/20 stratified train/test split.
7. **Evaluation**: accuracy, precision, recall, F1-score, and a confusion
   matrix, written to `model/evaluation_report.txt`.
8. **Serving**: `app.py` loads the saved model + vectorizer once, and the
   `/predict` endpoint runs the same `preprocess()` function used during
   training before handing the text to the model, so training and
   inference stay in sync.

## Extending this project

- Swap TF-IDF + Logistic Regression for word embeddings (Word2Vec,
  GloVe) with an LSTM/BiLSTM, or fine-tune a BERT-based classifier —
  `preprocess.py` and `train_model.py` are the two files you'd touch.
- Replace `data/reviews_dataset.csv` with a real, larger dataset for a
  more meaningful accuracy number.
- Add a `/history` endpoint and a small database if you want to log past
  predictions.
