"""
preprocess.py

Text cleaning and preprocessing shared by the training script and the
Flask app. Keeping this in one place means the exact same steps are
applied to training data and to whatever a user types into the app -
if the two pipelines drift apart the model's predictions stop making
sense, so this file is the single source of truth for it.
"""

import re

# Standard English stopwords, but deliberately missing negation words
# like "not", "no", "never", "n't" - those carry a lot of the sentiment
# signal ("not good" is the opposite of "good"), so removing them would
# throw away useful information.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "as", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "than", "too", "very", "s", "t", "can",
    "will", "just", "should", "now", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "this", "that", "these", "those", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their",
}

# A short list of suffix rules for a lightweight, rule based stemmer.
# This is nowhere near as thorough as something like the Porter stemmer,
# but it is dependency free and good enough to collapse common word
# forms (playing/played/plays -> play) for a project like this one.
_SUFFIX_RULES = [
    ("ies", "y"),
    ("sses", "ss"),
    ("ing", ""),
    ("edly", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text):
    """Lowercase the text and strip out urls, html, punctuation and digits."""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(text):
    return text.split()


def remove_stopwords(tokens):
    return [tok for tok in tokens if tok not in STOPWORDS]


def simple_stem(word):
    """Chop common suffixes off a word using a short list of rules."""
    if len(word) <= 4:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def preprocess(text, use_stemming=True):
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> stem.
    Returns a single space joined string, ready to hand to a vectorizer.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = [simple_stem(tok) for tok in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "I LOVE this product!!! 😊 #amazing",
        "The movie was okay, nothing special.",
        "I do not like this at all, it was terrible.",
    ]
    for s in samples:
        print(f"{s!r} -> {preprocess(s)!r}")
