"""
Multi-label evaluation metrics (spec item 17).

Written to be reusable: Stage B's CNN should be evaluated with exactly these
functions so the comparison in experiment 2 is like for like.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, hamming_loss, precision_recall_fscore_support,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def multilabel_metrics(Y_true, Y_pred, Y_score=None):
    """Per-label and aggregate metrics for a multi-label problem."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    p, r, f, sup = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0)

    per_label = {}
    for i, a in enumerate(C.ARTIFACTS):
        d = dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                 support=int(sup[i]))
        if Y_score is not None:
            try:
                d["auroc"] = float(roc_auc_score(Y_true[:, i], np.asarray(Y_score)[:, i]))
            except ValueError:
                d["auroc"] = float("nan")
        per_label[a] = d

    out = dict(
        per_label=per_label,
        macro_f1=float(np.mean(f)),
        micro_f1=float(precision_recall_fscore_support(
            Y_true, Y_pred, average="micro", zero_division=0)[2]),
        exact_match=float((Y_true == Y_pred).all(axis=1).mean()),
        hamming_loss=float(hamming_loss(Y_true, Y_pred)),
        n=int(len(Y_true)),
    )
    if Y_score is not None:
        aurocs = [per_label[a].get("auroc", np.nan) for a in C.ARTIFACTS]
        out["macro_auroc"] = float(np.nanmean(aurocs))
    return out


def by_n_artifacts(Y_true, Y_pred):
    """Exact-match accuracy as a function of how many artifacts are present."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    n = Y_true.sum(axis=1)
    rows = []
    for k in range(5):
        sel = n == k
        if sel.sum():
            rows.append(dict(n_artifacts=k, n_samples=int(sel.sum()),
                             exact_match=float((Y_true[sel] == Y_pred[sel]).all(axis=1).mean()),
                             hamming=float(hamming_loss(Y_true[sel], Y_pred[sel]))))
    return pd.DataFrame(rows)


def per_combination(Y_true, Y_pred):
    """Accuracy for each of the 16 label combinations (spec item 16)."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    bits = ["".join(map(str, row)) for row in Y_true]
    ok = (Y_true == Y_pred).all(axis=1)
    df = pd.DataFrame(dict(combo=bits, correct=ok))
    g = df.groupby("combo")["correct"].agg(["size", "mean"]).reset_index()
    g.columns = ["combo", "n_samples", "exact_match"]
    g["n_artifacts"] = g["combo"].str.count("1")
    return g.sort_values(["n_artifacts", "combo"]).reset_index(drop=True)


def label_confusions(Y_true, Y_pred):
    """2x2 confusion matrix per label."""
    return {a: confusion_matrix(np.asarray(Y_true)[:, i], np.asarray(Y_pred)[:, i],
                                labels=[0, 1])
            for i, a in enumerate(C.ARTIFACTS)}


def cross_label_confusion(Y_true, Y_pred):
    """
    For the look-alike pair analysis (spec item 18, experiment 3).

    Rate at which artifact j is falsely predicted when artifact i is the only
    one of the pair actually present.
    """
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    rows = []
    for i, ai in enumerate(C.ARTIFACTS):
        for j, aj in enumerate(C.ARTIFACTS):
            if i == j:
                continue
            sel = (Y_true[:, i] == 1) & (Y_true[:, j] == 0)
            if sel.sum() < 5:
                continue
            rows.append(dict(present=ai, falsely_predicted=aj, n=int(sel.sum()),
                             false_rate=float(Y_pred[sel, j].mean()),
                             recall_of_present=float(Y_pred[sel, i].mean())))
    if not rows:
        # e.g. a clean-only subset: no artifact is present anywhere, so there is
        # no look-alike pair to score.  Return the empty frame with its columns
        # rather than crashing on sort_values.
        return pd.DataFrame(columns=["present", "falsely_predicted", "n",
                                     "false_rate", "recall_of_present"])
    return pd.DataFrame(rows).sort_values("false_rate", ascending=False).reset_index(drop=True)


def to_markdown_table(metrics, title):
    lines = [f"### {title}", "",
             "| artifact | precision | recall | F1 | AUROC | support |",
             "|---|---|---|---|---|---|"]
    for a in C.ARTIFACTS:
        d = metrics["per_label"][a]
        auc = f"{d['auroc']:.3f}" if "auroc" in d else "-"
        lines.append(f"| {a} | {d['precision']:.3f} | {d['recall']:.3f} | "
                     f"{d['f1']:.3f} | {auc} | {d['support']} |")
    lines += ["", f"- macro-F1 **{metrics['macro_f1']:.4f}**",
              f"- micro-F1 {metrics['micro_f1']:.4f}",
              f"- exact-match accuracy **{metrics['exact_match']:.4f}**",
              f"- Hamming loss {metrics['hamming_loss']:.4f}"]
    if "macro_auroc" in metrics:
        lines.append(f"- macro-AUROC {metrics['macro_auroc']:.4f}")
    lines.append(f"- n = {metrics['n']}")
    return "\n".join(lines) + "\n"
