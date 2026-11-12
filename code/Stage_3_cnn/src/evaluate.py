"""
Evaluation.  Reuses Stage A's metrics module so experiment 2 compares the CNN
and the classical baseline through IDENTICAL code.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

def _load_stage_a_metrics():
    """
    Import Stage A's metrics.py by explicit file path.

    Deliberately NOT via sys.path: Stage A has its own config.py and plots.py,
    and putting its folder on the path shadows Stage B's modules of the same
    name (this bit once - Stage B's plots.py silently resolved to Stage A's).
    """
    import importlib.util
    path = C.STAGE_A / "src" / "metrics.py"
    if not path.exists():
        raise SystemExit(f"Stage A metrics not found at {path}. Run Stage A first, "
                         "or fix STAGE_A in config.py.")
    sys.path.insert(0, str(C.STAGE_A))          # metrics.py imports Stage A's config
    try:
        spec = importlib.util.spec_from_file_location("stage_a_metrics", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(C.STAGE_A))
    return mod


M = _load_stage_a_metrics()


def tune_thresholds(P_val, Y_val):
    """Per-label decision thresholds maximising validation F1 (never test)."""
    thr = []
    for i in range(Y_val.shape[1]):
        best = (-1.0, 0.5)
        for t in C.THRESHOLD_GRID:
            f = f1_score(Y_val[:, i], (P_val[:, i] >= t).astype(int), zero_division=0)
            if f > best[0]:
                best = (f, t)
        thr.append(best[1])
    return np.array(thr)


def apply(P, thr):
    return (P >= thr[None, :]).astype(int)


def severity_errors(S_pred, S_true, Y_true):
    """MAE / RMSE per artifact, computed only where the artifact is present."""
    out = {}
    for i, a in enumerate(C.ARTIFACTS):
        sel = Y_true[:, i] == 1
        if sel.sum() < 5:
            continue
        err = S_pred[sel, i] - S_true[sel, i]
        out[a] = dict(mae=float(np.abs(err).mean()),
                      rmse=float(np.sqrt((err ** 2).mean())), n=int(sel.sum()))
    return out


def full_report(P, Y, thr):
    pred = apply(P, thr)
    rep = M.multilabel_metrics(Y, pred, P)
    rep["by_n_artifacts"] = M.by_n_artifacts(Y, pred).to_dict("records")
    rep["per_combination"] = M.per_combination(Y, pred).to_dict("records")
    rep["cross_label_confusion"] = M.cross_label_confusion(Y, pred).to_dict("records")
    return rep, pred


def bh_scatter_pair(cross_records):
    """Pull out the look-alike pair rates for experiment 3."""
    d = {}
    for r in cross_records:
        if r["present"] == "scatter" and r["falsely_predicted"] == "beam_hardening":
            d["scatter_to_bh"] = r["false_rate"]
        if r["present"] == "beam_hardening" and r["falsely_predicted"] == "scatter":
            d["bh_to_scatter"] = r["false_rate"]
    return d
