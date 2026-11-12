"""
Rule-based threshold classifier (spec item 19).

"a fully non-DL baseline reflecting how QA engineers currently flag artifacts
heuristically" -- one feature, one threshold, per artifact.

Rather than hard-coding which feature belongs to which artifact, each rule
SEARCHES all 14 features and both directions and keeps whichever single
threshold maximises F1 on the VALIDATION split.  That is fairer than assuming
the mapping, and the features it picks are themselves a result: if beam
hardening and scatter both select a cupping-type feature, that is evidence for
the confusion analysis in experiment 3 rather than an assumption baked in.

Candidate thresholds come from quantiles of the TRAIN distribution; selection
is on validation; the test split is never touched.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


class SingleFeatureRule:
    """One feature, one threshold, one direction."""

    def __init__(self, n_thresholds=120):
        self.n_thresholds = n_thresholds
        self.feature_index = None
        self.feature_name = None
        self.threshold = None
        self.direction = None          # '>' means predict 1 when x > threshold
        self.val_f1 = None

    def fit(self, X_train, X_val, y_val, feature_names):
        best = (-1.0, None, None, None)
        qs = np.linspace(0.01, 0.99, self.n_thresholds)
        for j in range(X_train.shape[1]):
            cands = np.unique(np.quantile(X_train[:, j], qs))
            for t in cands:
                for direction in (">", "<"):
                    pred = (X_val[:, j] > t) if direction == ">" else (X_val[:, j] < t)
                    f1 = f1_score(y_val, pred.astype(int), zero_division=0)
                    if f1 > best[0]:
                        best = (f1, j, float(t), direction)
        self.val_f1, self.feature_index, self.threshold, self.direction = best
        self.feature_name = feature_names[self.feature_index]
        return self

    def predict(self, X):
        x = X[:, self.feature_index]
        pred = (x > self.threshold) if self.direction == ">" else (x < self.threshold)
        return pred.astype(int)

    def score_like(self, X):
        """A pseudo-score for AUROC: signed distance from the threshold."""
        x = X[:, self.feature_index]
        d = x - self.threshold
        return d if self.direction == ">" else -d

    def describe(self):
        return f"{self.feature_name} {self.direction} {self.threshold:.5g}"


class RuleBasedMultiLabel:
    """One SingleFeatureRule per artifact."""

    def __init__(self):
        self.rules = {}

    def fit(self, X_train, X_val, Y_val, feature_names):
        for i, a in enumerate(C.ARTIFACTS):
            self.rules[a] = SingleFeatureRule().fit(X_train, X_val, Y_val[:, i],
                                                    feature_names)
        return self

    def predict(self, X):
        return np.stack([self.rules[a].predict(X) for a in C.ARTIFACTS], axis=1)

    def scores(self, X):
        return np.stack([self.rules[a].score_like(X) for a in C.ARTIFACTS], axis=1)

    def summary(self):
        return {a: dict(rule=r.describe(), feature=r.feature_name,
                        threshold=r.threshold, direction=r.direction,
                        val_f1=round(float(r.val_f1), 4))
                for a, r in self.rules.items()}
