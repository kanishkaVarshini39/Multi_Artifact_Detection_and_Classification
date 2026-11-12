"""
Random Forest and XGBoost baselines (spec item 19), one binary model per label.

Protocol, kept strict so the test split stays clean:
  1. hyperparameters selected on VALIDATION by AUROC (threshold-free)
  2. decision threshold tuned separately on VALIDATION by F1
  3. test touched exactly once, at the end
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def _grid(d):
    keys = list(d)
    for combo in itertools.product(*(d[k] for k in keys)):
        yield dict(zip(keys, combo))


def _make(kind, params):
    if kind == "rf":
        return RandomForestClassifier(random_state=C.SEED, n_jobs=-1, **params)
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=C.SEED, n_jobs=-1, tree_method="hist",
                         eval_metric="logloss", **params)


def tune_one_label(kind, X_tr, y_tr, X_val, y_val):
    """Return (fitted_model, params, val_auroc)."""
    grid = C.RF_GRID if kind == "rf" else C.XGB_GRID
    best = (-1.0, None, None)
    for params in _grid(grid):
        m = _make(kind, params).fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if auc > best[0]:
            best = (auc, m, params)
    return best[1], best[2], best[0]


def tune_threshold(y_val, p_val):
    """Decision threshold maximising validation F1."""
    best = (-1.0, 0.5)
    for t in C.THRESHOLD_GRID:
        f1 = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f1 > best[0]:
            best = (f1, t)
    return best[1], best[0]


class TunedMultiLabel:
    """Four independently tuned binary models with per-label thresholds."""

    def __init__(self, kind):
        self.kind = kind
        self.models, self.params, self.thresholds, self.val_auroc = {}, {}, {}, {}

    def fit(self, X_tr, Y_tr, X_val, Y_val, verbose=True):
        for i, a in enumerate(C.ARTIFACTS):
            m, params, auc = tune_one_label(self.kind, X_tr, Y_tr[:, i], X_val, Y_val[:, i])
            p_val = m.predict_proba(X_val)[:, 1]
            thr, f1 = tune_threshold(Y_val[:, i], p_val)
            self.models[a], self.params[a] = m, params
            self.thresholds[a], self.val_auroc[a] = thr, auc
            if verbose:
                print(f"    {a:15s} val AUROC={auc:.4f}  thr={thr:.2f}  "
                      f"val F1={f1:.4f}  {params}")
        return self

    def predict_proba(self, X):
        return np.stack([self.models[a].predict_proba(X)[:, 1] for a in C.ARTIFACTS], axis=1)

    def predict(self, X):
        P = self.predict_proba(X)
        thr = np.array([self.thresholds[a] for a in C.ARTIFACTS])
        return (P >= thr).astype(int)

    def summary(self):
        return {a: dict(params=self.params[a], threshold=self.thresholds[a],
                        val_auroc=round(float(self.val_auroc[a]), 4))
                for a in C.ARTIFACTS}
