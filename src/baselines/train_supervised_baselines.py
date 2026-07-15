"""Phase 4: reproduce the paper's 10 supervised ML/DL baseline classifiers
(external/Code-for-ICS-NAD-Dataset/Train_All_in_One_czh.ipynb: GaussianNB, ANN, KNN,
RandomForest, GBDT, XGBoost, LightGBM, SVC, AdaBoost, ELM) on the harmonized Parquet
corpus, using the exact same file-level/intra-file split as the RL arms (Phase 2/3) —
the paper's own reported numbers used a naive random row-level split, which leaks flows
from the same capture session across train/test; this is the fair, leak-free comparison.

Binary target (state != BENIGN) to match the RL environment's is_attack framing.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from baselines.data import load_split
from baselines.models import ANNClassifier, ELMClassifier

RESULTS_PATH = Path("/home/user5/Desktop/MS THESIS/data/supervised_baseline_results.json")
MODELS_DIR = Path("/home/user5/Desktop/MS THESIS/data/baseline_models")

TRAIN_ROWS = 300_000
EVAL_ROWS = 50_000
SVC_TRAIN_ROWS = 15_000  # SVC scales poorly (O(n^2)-O(n^3)); subsampled further, documented
KNN_TRAIN_ROWS = 50_000  # KNN inference cost scales with train set size


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():
    print("Loading data...")
    X_train, y_train = load_split("train", max_rows=TRAIN_ROWS)
    X_val, y_val = load_split("val", max_rows=EVAL_ROWS)
    X_test, y_test = load_split("test", max_rows=EVAL_ROWS)
    print(f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"train positive rate={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    models = {
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "GBDT": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
        "SVC": SVC(kernel="rbf", random_state=42, probability=True),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "ANN": ANNClassifier(input_dim=X_train_s.shape[1]),
        "ELM": ELMClassifier(hidden_units=200),
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        if name == "SVC":
            Xtr, ytr = X_train_s[:SVC_TRAIN_ROWS], y_train[:SVC_TRAIN_ROWS]
        elif name == "KNN":
            Xtr, ytr = X_train_s[:KNN_TRAIN_ROWS], y_train[:KNN_TRAIN_ROWS]
        else:
            Xtr, ytr = X_train_s, y_train

        t0 = time.time()
        model.fit(Xtr, ytr)
        train_time = time.time() - t0

        val_pred = model.predict(X_val_s)
        test_pred = model.predict(X_test_s)

        val_metrics = evaluate(y_val, val_pred)
        test_metrics = evaluate(y_test, test_pred)
        results[name] = {
            "train_time_sec": train_time,
            "train_rows_used": len(Xtr),
            "val": val_metrics,
            "test": test_metrics,
        }
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"train_time={train_time:.1f}s  val_f1={val_metrics['f1']:.3f}  test_f1={test_metrics['f1']:.3f}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'model':15s} {'train_s':>8s} {'val_f1':>8s} {'val_prec':>9s} {'val_rec':>8s} {'test_f1':>8s} {'test_prec':>10s} {'test_rec':>9s}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["test"]["f1"]):
        print(f"{name:15s} {r['train_time_sec']:8.1f} {r['val']['f1']:8.3f} {r['val']['precision']:9.3f} "
              f"{r['val']['recall']:8.3f} {r['test']['f1']:8.3f} {r['test']['precision']:10.3f} {r['test']['recall']:9.3f}")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
