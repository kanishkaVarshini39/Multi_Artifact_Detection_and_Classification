"""Feature-importance analysis -- the setup for experiment 3."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def permutation_importances(tuned, X_val, Y_val, n_repeats=10):
    """
    Permutation importance (AUROC drop) per artifact x feature, measured on the
    validation split.

    This is the key diagnostic for the beam-hardening / scatter confusion: if
    the two labels depend on the SAME features, the look-alike problem is
    structural rather than incidental, and experiment 3 has a mechanism to
    report rather than just a number.
    """
    rows = []
    for i, a in enumerate(C.ARTIFACTS):
        r = permutation_importance(tuned.models[a], X_val, Y_val[:, i],
                                   scoring="roc_auc", n_repeats=n_repeats,
                                   random_state=C.SEED, n_jobs=1)
        for j, f in enumerate(C.FEATURE_NAMES):
            rows.append(dict(artifact=a, feature=f,
                             importance=float(r.importances_mean[j]),
                             std=float(r.importances_std[j])))
    return pd.DataFrame(rows)


def importance_matrix(df):
    """artifact x feature table, each row normalised to sum to 1."""
    m = df.pivot(index="artifact", columns="feature", values="importance")
    m = m.reindex(index=list(C.ARTIFACTS), columns=C.FEATURE_NAMES)
    m = m.clip(lower=0)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def feature_overlap(matrix):
    """
    Cosine similarity between each pair of artifacts' importance profiles.
    High similarity = the two labels are read from the same evidence.
    """
    A = matrix.to_numpy()
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    S = (A / norm) @ (A / norm).T
    return pd.DataFrame(S, index=matrix.index, columns=matrix.index)


def top_features(df, k=3):
    out = {}
    for a in C.ARTIFACTS:
        sub = df[df.artifact == a].nlargest(k, "importance")
        out[a] = [(r.feature, round(r.importance, 4)) for r in sub.itertuples()]
    return out
