"""Tasks 3 + 4 - train TF-IDF + logistic regression variants, log each to MLflow,
pick the best by cross-validated macro-F1 and save it for the API.

Usage:  python -m invoice_ml.train
UI:     mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from invoice_ml.model import CONFIGS, MODELS, ROOT, SEED, build_pipeline, load_data

EXPERIMENT = "invoice-line-category"


def log_sklearn_model(pipe) -> None:
    """Log the pipeline; works on MLflow 2.x and 3.x (3.x uses skops and must be
    told our own text normaliser is a trusted type)."""
    trusted = ["invoice_ml.text_utils.normalize_text"]
    try:
        mlflow.sklearn.log_model(pipe, name="model", skops_trusted_types=trusted)
    except TypeError:
        mlflow.sklearn.log_model(pipe, artifact_path="model")


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.25, stratify=df["label"], random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = []
    for name, cfg in CONFIGS.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(cfg)
            cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            metrics = {
                "cv_macro_f1": float(cv_f1.mean()),
                "cv_macro_f1_std": float(cv_f1.std()),
                "test_macro_f1": f1_score(y_te, pred, average="macro"),
                "test_accuracy": accuracy_score(y_te, pred),
                "n_train": len(X_tr), "n_test": len(X_te),
            }
            mlflow.log_params({k: str(v) for k, v in cfg.items()} |
                              {"model": "LogisticRegression", "class_weight": "balanced",
                               "test_size": 0.25, "seed": SEED})
            mlflow.log_metrics(metrics)
            log_sklearn_model(pipe)
            results.append({"name": name, "run_id": run.info.run_id, "cfg": cfg, **metrics})
            print(f"{name:14} cv_macro_f1={metrics['cv_macro_f1']:.3f}"
                  f"±{metrics['cv_macro_f1_std']:.3f}  test_macro_f1={metrics['test_macro_f1']:.3f}"
                  f"  test_acc={metrics['test_accuracy']:.3f}")

    # Select on CV (train only) so the test set stays an unbiased final check;
    # ties -> the simpler model (earlier in CONFIGS).
    best = max(results, key=lambda r: round(r["cv_macro_f1"], 3))
    print(f"\nSelected: {best['name']} (run {best['run_id']})")

    final = build_pipeline(best["cfg"]).fit(X_tr, y_tr)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "model.joblib")
    (MODELS / "best_run.json").write_text(json.dumps(
        {k: v for k, v in best.items()}, indent=2, default=list), encoding="utf-8")
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected", "true")


if __name__ == "__main__":
    main()
