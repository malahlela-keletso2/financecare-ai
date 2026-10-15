from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from financecare_ai import (BASE_DIR, BOT_NAME, DEFAULT_CSV, DEFAULT_MODEL, build_features, file_digest, make_labels,
                               read_dataframe)

SEED = 42


def candidates(seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=250,
                                                                 class_weight="balanced", random_state=seed),
    }


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def main() -> int:
    ap = argparse.ArgumentParser(description=f"{BOT_NAME}: train the distress model")
    ap.add_argument("--data", default=DEFAULT_CSV)
    ap.add_argument("--outcomes", default=None, help="optional CSV: Customer ID, distress_outcome (0/1)")
    ap.add_argument("--model-out", default=DEFAULT_MODEL)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import joblib
    import sklearn
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

    data_path = Path(args.data) if Path(args.data).exists() else BASE_DIR / args.data
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")
    model_path = Path(args.model_out) if Path(args.model_out).is_absolute() else BASE_DIR / args.model_out
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_dataframe(str(data_path))
    X = build_features(df)
    if args.outcomes:
        out = pd.read_csv(args.outcomes).set_index("Customer ID")["distress_outcome"]
        y = df["Customer ID"].map(out)
        if y.isna().any():
            sys.exit(f"--outcomes is missing {int(y.isna().sum())} customers from the data file.")
        y, label_source = y.astype(int), f"real outcomes from {args.outcomes}"
    else:
        y, label_source = make_labels(df), "rule-engine flags (EARLY WARNING or worse)"
    ids = df["Customer ID"].astype(int)
    print(f"Rows: {len(df):,} | features: {X.shape[1]} | positives: {int(y.sum())} ({y.mean():.1%}) | labels: {label_source}")

    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, ids, test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"Train: {len(X_tr):,} rows | Test (held out, untouched): {len(X_te):,} rows")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    results, cv_proba = {}, {}
    for name, model in candidates(args.seed).items():
        proba = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        cv_proba[name] = proba
        results[name] = float(average_precision_score(y_tr, proba))
        print(f"  CV average precision  {name:<24s} {results[name]:.4f}")
    best = max(results, key=results.get)
    print(f"Selected: {best}")

    threshold = best_threshold(y_tr.to_numpy(), cv_proba[best])
    model = candidates(args.seed)[best].fit(X_tr, y_tr)
    print(f"Decision threshold (max F1 on cross-validated train predictions): {threshold:.3f}")

    bundle = dict(model=model, feature_columns=list(X.columns), threshold=threshold, meta=dict(
        trained_at=datetime.now().isoformat(timespec="seconds"), model_name=best, label_source=label_source,
        sklearn_version=sklearn.__version__, data_file=data_path.name, data_sha256=file_digest(str(data_path)),
        seed=args.seed, test_size=args.test_size, cv_average_precision=results,
        n_train=int(len(X_tr)), n_test=int(len(X_te))))
    joblib.dump(bundle, model_path)
    split = dict(seed=args.seed, label_source=label_source, data_sha256=bundle["meta"]["data_sha256"],
                 train_ids=sorted(map(int, id_tr)), test_ids=sorted(map(int, id_te)))
    (model_path.parent / "split.json").write_text(json.dumps(split))
    if args.outcomes:
        y.rename("label").to_frame().assign(**{"Customer ID": ids}).to_csv(model_path.parent / "labels.csv", index=False)
    print(f"Saved model -> {model_path}\nSaved split -> {model_path.parent / 'split.json'}")
    print("Next: python test_distress.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
