from __future__ import annotations

import argparse
import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import financecare_ai as bot_mod
from financecare_ai import (BASE_DIR, BOT_NAME, CONFIG, DEFAULT_CSV, DEFAULT_MODEL, Bot, Store, assess, build_features,
                               file_digest, make_labels, parse_router_reply, read_dataframe)


MIN_ROC_AUC, MIN_RECALL, MIN_PRECISION = 0.95, 0.85, 0.80


def record(**kw) -> dict:
    t = pd.Timestamp("2023-06-01")
    base = {"Customer ID": 1, "First Name": "Test", "Last Name": "User", "Contact Number": 12345678901,
            "Email": "t@example.com", "Address": "1 Main St", "City": "Town", "Anomaly": 1,
            "Account Balance": 5000.0, "Account Balance After Transaction": 5200.0,
            "Transaction Type": "Deposit", "Transaction Amount": 200.0, "Transaction Date": t,
            "Credit Limit": 10000.0, "Credit Card Balance": 1000.0, "Minimum Payment Due": 50.0,
            "Payment Due Date": t, "Last Credit Card Payment Date": t - timedelta(days=3),
            "Loan Status": "Rejected", "Loan Amount": 0.0, "Loan Type": "Personal"}
    base.update(kw)
    return base


class TestScoring(unittest.TestCase):
    def codes(self, **kw):
        return {s["code"] for s in assess(record(**kw))["signals"]}

    def test_healthy_customer_has_no_flag(self):
        a = assess(record())
        self.assertEqual((a["status"], a["score"], a["signals"]), ("NO FLAG", 0, []))

    def test_utilisation_points(self):
        pts = lambda ccb: next(s["points"] for s in assess(record(**{"Credit Card Balance": ccb}))["signals"]
                               if s["code"] == "HIGH_UTILISATION")
        self.assertEqual((pts(7500), pts(9000), pts(10001)), (1, 2, 3))
        self.assertNotIn("HIGH_UTILISATION", self.codes(**{"Credit Card Balance": 7400}))

    def test_late_payment_points(self):
        late = lambda d: assess(record(**{"Last Credit Card Payment Date": pd.Timestamp("2023-06-01") + timedelta(days=d)}))
        self.assertEqual(late(10)["signals"][0]["points"], 1)
        self.assertEqual(late(31)["signals"][0]["points"], 2)
        self.assertEqual(late(0)["signals"], [])

    def test_low_balance_and_overdrawn(self):
        self.assertIn("LOW_BALANCE", self.codes(**{"Account Balance": 900.0}))
        self.assertIn("OVERDRAWN", self.codes(**{"Account Balance After Transaction": -5.0}))
        self.assertIn("CANNOT_COVER_MIN", self.codes(**{"Account Balance": 20.0}))

    def test_loan_burden_only_counts_approved_loans(self):
        approved = self.codes(**{"Loan Status": "Approved", "Loan Amount": 60000.0})
        rejected = self.codes(**{"Loan Status": "Rejected", "Loan Amount": 60000.0})
        self.assertIn("LOAN_BURDEN", approved)
        self.assertNotIn("LOAN_BURDEN", rejected)

    def test_status_thresholds(self):
        early = assess(record(**{"Credit Card Balance": 10500.0, "Account Balance": 900.0,
                                 "Account Balance After Transaction": 900.0}))
        self.assertEqual((early["score"], early["status"]), (4, "EARLY WARNING"))
        dist = assess(record(**{"Credit Card Balance": 10500.0, "Account Balance": 100.0,
                                "Account Balance After Transaction": -10.0}))
        self.assertEqual(dist["status"], "FINANCIAL DISTRESS")

    def test_zero_credit_limit_does_not_crash(self):
        self.assertEqual(assess(record(**{"Credit Limit": 0.0}))["metrics"]["util"], 0.0)

    def test_demographics_never_change_the_result(self):
        a = assess(record(**{"Credit Card Balance": 9500.0}))
        b = assess(record(**{"Credit Card Balance": 9500.0, "Age": 80, "Gender": "Female", "City": "Elsewhere"}))
        self.assertEqual((a["status"], a["score"]), (b["status"], b["score"]))


class TestFeaturesAndRouter(unittest.TestCase):
    def test_features_exclude_personal_attributes(self):
        import re
        df = read_dataframe(str(DATA))
        words = set(re.findall(r"[a-z]+", " ".join(build_features(df.head(50)).columns).lower()))
        for banned in ("age", "gender", "city", "address", "email", "first", "last", "name", "contact", "customer", "id"):
            self.assertNotIn(banned, words, f"personal attribute '{banned}' leaked into model features")

    def test_features_are_numeric_and_finite(self):
        X = build_features(read_dataframe(str(DATA)).head(200))
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertTrue(all(np.issubdtype(t, np.number) for t in X.dtypes))

    def test_router_reply_parsing(self):
        self.assertEqual(parse_router_reply('{"index": 2, "customer_id": null}'), (2, None))
        self.assertEqual(parse_router_reply('noise {"index": 1, "customer_id": 77} noise'), (1, 77))
        self.assertEqual(parse_router_reply('{"index": 999, "customer_id": true}'), (None, None))
        self.assertEqual(parse_router_reply("not json"), (None, None))


class TestBotWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [record(**{"Customer ID": 1, "Credit Card Balance": 10500.0, "Account Balance": 900.0}),
                record(**{"Customer ID": 2, "Credit Card Balance": 10500.0, "Account Balance": 100.0,
                          "Account Balance After Transaction": -10.0}),
                record(**{"Customer ID": 3})]
        cls.assessments = {r["Customer ID"]: assess(r) for r in rows}

    def setUp(self):
        self.bot = Bot(self.assessments, Store(":memory:"))
        self.s = self.bot.new_session()

    def test_check_and_not_found(self):
        self.assertIn("EARLY WARNING", self.bot.handle("Check customer 1", self.s))
        self.assertIn("was not found", self.bot.handle("Check customer 999", self.s))

    def test_flag_lists_sorted_by_priority(self):
        out = self.bot.handle("Find customers at risk", self.s)
        self.assertLess(out.index("Customer 2"), out.index("Customer 1"))
        self.assertNotIn("Customer 3", out)

    def test_resolve_requires_note_and_hides_flag(self):
        self.assertIn("short review note", self.bot.handle("resolve flag for 1", self.s))
        self.assertIn("marked as resolved", self.bot.handle("resolve flag for 1: called customer, plan agreed", self.s))
        self.assertNotIn("Customer 1 ", self.bot.handle("Find customers at risk", self.s))

    def test_distress_needs_a_different_supervisor(self):
        self.assertIn("supervisor approval", self.bot.handle("resolve flag for 2: spoke with customer today", self.s))
        self.assertIn("valid supervisor", self.bot.handle("approve 2 by E1234", self.s))
        self.assertIn("approved by SUP002", self.bot.handle("approve 2 by SUP002", self.s))

    def test_resolution_expires_if_customer_worsens(self):
        self.bot.handle("resolve flag for 1: called customer, plan agreed", self.s)
        worse = dict(self.assessments[1], status="FINANCIAL DISTRESS", score=8)
        bot2 = Bot({**self.assessments, 1: worse}, self.bot.store)
        self.assertIn("Customer 1", bot2.handle("Find customers at risk", bot2.new_session()))

    def test_audit_log_written(self):
        self.bot.handle("Check customer 3", self.s)
        self.assertEqual(self.bot.store.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0], 1)


def evaluate_model(data_path: Path, model_path: Path, out_dir: Path) -> dict:
    import joblib
    from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
                                 classification_report, confusion_matrix, f1_score, precision_score, recall_score,
                                 roc_auc_score)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = joblib.load(model_path)
    split = json.loads((model_path.parent / "split.json").read_text())
    assert split["data_sha256"] == file_digest(str(data_path)), "Data file changed since training - retrain first."
    assert not set(split["train_ids"]) & set(split["test_ids"]), "Train/test leakage: IDs overlap."

    df = read_dataframe(str(data_path))
    test = df[df["Customer ID"].astype(int).isin(split["test_ids"])].reset_index(drop=True)
    labels_file = model_path.parent / "labels.csv"
    if labels_file.exists():
        lab = pd.read_csv(labels_file).set_index("Customer ID")["label"]
        y = test["Customer ID"].map(lab).astype(int)
    else:
        y = make_labels(test)
    X = build_features(test).reindex(columns=bundle["feature_columns"], fill_value=0.0)
    proba = bundle["model"].predict_proba(X)[:, 1]
    pred = (proba >= bundle["threshold"]).astype(int)

    rep = dict(n_test=int(len(y)), positives=int(y.sum()), threshold=round(bundle["threshold"], 4),
               model=bundle["meta"]["model_name"], label_source=bundle["meta"]["label_source"],
               roc_auc=float(roc_auc_score(y, proba)), average_precision=float(average_precision_score(y, proba)),
               precision=float(precision_score(y, pred)), recall=float(recall_score(y, pred)), f1=float(f1_score(y, pred)),
               confusion_matrix=confusion_matrix(y, pred).tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_report.json").write_text(json.dumps(rep, indent=2))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    RocCurveDisplay.from_predictions(y, proba, ax=ax[0]); ax[0].set_title("ROC curve (held-out test set)")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax[1]); ax[1].set_title("Precision-recall curve")
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["No flag", "Flagged"], ax=ax[2], colorbar=False)
    ax[2].set_title(f"Confusion matrix (threshold {bundle['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(out_dir / "test_evaluation.png", dpi=130); plt.close(fig)

    inner = bundle["model"]
    imp = None
    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(inner, X, y, n_repeats=5, random_state=0, scoring="average_precision").importances_mean
    top = pd.Series(imp, index=bundle["feature_columns"]).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5)); top.plot.barh(ax=ax); ax.set_title("Top feature importances")
    fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=130); plt.close(fig)

    print("\n" + "=" * 70 + "\nHELD-OUT TEST SET EVALUATION\n" + "=" * 70)
    print(f"Model: {rep['model']} | labels: {rep['label_source']}\nTest rows: {rep['n_test']} (positives: {rep['positives']})")
    print(classification_report(y, pred, target_names=["No flag", "Flagged"], digits=3))
    print(f"ROC-AUC {rep['roc_auc']:.4f} | Average precision {rep['average_precision']:.4f}")
    print(f"Saved: {out_dir / 'test_report.json'}, test_evaluation.png, feature_importance.png")
    return rep


class TestHeldOutModel(unittest.TestCase):
    report: dict = {}

    def test_model_quality_on_unseen_customers(self):
        r = type(self).report
        self.assertGreaterEqual(r["roc_auc"], MIN_ROC_AUC)
        self.assertGreaterEqual(r["recall"], MIN_RECALL, "Model misses too many flagged customers.")
        self.assertGreaterEqual(r["precision"], MIN_PRECISION, "Model raises too many false alarms.")


DATA = BASE_DIR / DEFAULT_CSV


def main() -> int:
    global DATA
    ap = argparse.ArgumentParser(description=f"{BOT_NAME}: unit tests and held-out model evaluation")
    ap.add_argument("--data", default=DEFAULT_CSV)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--skip-model", action="store_true", help="run unit tests only")
    args = ap.parse_args()
    DATA = Path(args.data) if Path(args.data).exists() else BASE_DIR / args.data
    model_path = Path(args.model) if Path(args.model).is_absolute() else BASE_DIR / args.model

    suite = unittest.TestSuite()
    load = unittest.defaultTestLoader.loadTestsFromTestCase
    for c in (TestScoring, TestFeaturesAndRouter, TestBotWorkflow):
        suite.addTests(load(c))
    if not args.skip_model:
        if not model_path.exists():
            print(f"No trained model at {model_path} - run `python train_model.py` first (or use --skip-model).")
            return 2
        TestHeldOutModel.report = evaluate_model(DATA, model_path, BASE_DIR / args.reports)
        suite.addTests(load(TestHeldOutModel))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
