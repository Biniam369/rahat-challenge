"""Task 5 - per-class precision/recall/F1, confusion matrix and model mistakes.

The test split is small (~18 lines), so we report two views:
  * held-out test set (same split as train.py) - the honest final number
  * 5-fold out-of-fold predictions over ALL lines - more stable, and gives
    enough errors to analyse
Usage:  python -m invoice_ml.evaluate
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from invoice_ml.model import LABELS, MODELS, ROOT, SEED, build_pipeline, load_data


def cm_table(y_true, y_pred) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    df = pd.DataFrame(cm, index=[f"true:{l}" for l in LABELS], columns=LABELS)
    return "```\n" + df.to_string() + "\n```"


def main() -> None:
    best = json.loads((MODELS / "best_run.json").read_text(encoding="utf-8"))
    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.25, stratify=df["label"], random_state=SEED)

    pipe = build_pipeline(best["cfg"]).fit(X_tr, y_tr)
    test_pred = pipe.predict(X_te)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build_pipeline(best["cfg"]), df["text"], df["label"], cv=cv)
    proba = cross_val_predict(build_pipeline(best["cfg"]), df["text"], df["label"], cv=cv,
                              method="predict_proba")
    df = df.assign(pred=oof, confidence=proba.max(axis=1))
    mistakes = df[df["label"] != df["pred"]].sort_values("confidence", ascending=False)

    out = [f"# Evaluation - `{best['name']}`\n",
           "## Held-out test set\n", "```",
           classification_report(y_te, test_pred, labels=LABELS, zero_division=0), "```",
           cm_table(y_te, test_pred), "",
           "## 5-fold out-of-fold (all lines)\n", "```",
           classification_report(df["label"], df["pred"], labels=LABELS, zero_division=0), "```",
           cm_table(df["label"], df["pred"]), "",
           f"## Mistakes ({len(mistakes)}) - most confident first\n",
           "| text | true | predicted | confidence |", "|---|---|---|---|"]
    out += [f"| {r.text} | {r.label} | {r.pred} | {r.confidence:.2f} |"
            for r in mistakes.itertuples()]
    report = "\n".join(out)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "evaluation.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
