"""Model definitions shared by training and evaluation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from invoice_ml.text_utils import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "line_items_labelled.csv"
MODELS = ROOT / "models"
LABELS = ["cement", "steel", "aggregates", "blocks", "services"]
SEED = 42

# Each config = one MLflow run. Compared on cross-validated macro-F1.
CONFIGS = {
    "word_1-2gram": {"features": "word", "word_ngram": (1, 2), "C": 10.0},
    "char_2-5gram": {"features": "char", "char_ngram": (2, 5), "C": 10.0},
    "word+char": {"features": "word+char", "word_ngram": (1, 2), "char_ngram": (2, 5), "C": 10.0},
}


def _word(ngram):
    return TfidfVectorizer(analyzer="word", ngram_range=tuple(ngram), preprocessor=normalize_text,
                           token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)


def _char(ngram):
    # char_wb n-grams cope with Arabic prefixes (ال, بال), plurals, typos and
    # tokens like "T12"/"50kg" that a word tokenizer splits badly.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(ngram),
                           preprocessor=normalize_text, sublinear_tf=True)


def build_pipeline(cfg: dict) -> Pipeline:
    if cfg["features"] == "word":
        feats = _word(cfg["word_ngram"])
    elif cfg["features"] == "char":
        feats = _char(cfg["char_ngram"])
    else:
        feats = FeatureUnion([("word", _word(cfg["word_ngram"])),
                              ("char", _char(cfg["char_ngram"]))])
    clf = LogisticRegression(C=cfg["C"], max_iter=2000, class_weight="balanced",
                             random_state=SEED)
    return Pipeline([("tfidf", feats), ("clf", clf)])


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8")
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"] != "out_of_scope"].reset_index(drop=True)  # no class for it (see guide)
    bad = set(df["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"Unknown labels in data: {bad}")
    return df
