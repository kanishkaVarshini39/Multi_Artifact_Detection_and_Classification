"""
Stage A - Classical baseline for Project 29.

Builds the three non-deep-learning baselines the brief asks for (item 19),
evaluates them with the full multi-label metric set (item 17), and produces the
feature-importance analysis that sets up the beam-hardening / scatter confusion
study (item 18, experiment 3).

    python3 run_baseline.py

Everything is written to results/.  Runs in a few minutes on CPU; no GPU and no
image files are needed, only labels.csv and features.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import config as C
from data import load, describe
import metrics as M
import analysis as A
import plots as P
from rule_based import RuleBasedMultiLabel
from models import TunedMultiLabel


def main():
    t0 = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}

    print("=" * 78)
    print("STAGE A - CLASSICAL BASELINE")
    print("=" * 78)

    df, splits, root, layout = load()
    print(f"\ndataset: {root}  ({layout} layout)")
    print(describe(splits))
    for s in ("train", "val", "test"):
        if s not in splits:
            raise SystemExit(f"missing required split: {s}")

    Xtr, Ytr = splits["train"]["X"], splits["train"]["Y"]
    Xva, Yva = splits["val"]["X"], splits["val"]["Y"]
    Xte, Yte = splits["test"]["X"], splits["test"]["Y"]

    results, preds = {}, {}

    # ---------------------------------------------------------------- 1. rules
    print("\n" + "-" * 78)
    print("1. Rule-based single-feature thresholds (how QA is done today)")
    print("-" * 78)
    rules = RuleBasedMultiLabel().fit(Xtr, Xva, Yva, C.FEATURE_NAMES)
    for a, d in rules.summary().items():
        print(f"    {a:15s} {d['rule']:42s} val F1={d['val_f1']:.4f}")
    m = M.multilabel_metrics(Yte, rules.predict(Xte), rules.scores(Xte))
    results["rule-based"] = m
    preds["rule-based"] = rules.predict(Xte)
    out["rule_based"] = dict(rules=rules.summary(), test=m)
    print(f"    -> test macro-F1 {m['macro_f1']:.4f}   exact-match {m['exact_match']:.4f}")

    # ------------------------------------------------------- 2. trees
    tuned = {}
    for kind, label in [("rf", "Random Forest"), ("xgb", "XGBoost")]:
        print("\n" + "-" * 78)
        print(f"{2 if kind == 'rf' else 3}. {label} on all 14 features")
        print("-" * 78)
        try:
            t = TunedMultiLabel(kind).fit(Xtr, Ytr, Xva, Yva)
        except ImportError:
            print("    xgboost not installed - skipped")
            continue
        m = M.multilabel_metrics(Yte, t.predict(Xte), t.predict_proba(Xte))
        tuned[kind] = t
        results[label] = m
        preds[label] = t.predict(Xte)
        out[kind] = dict(selection=t.summary(), test=m)
        print(f"    -> test macro-F1 {m['macro_f1']:.4f}   exact-match {m['exact_match']:.4f}")

    # ------------------------------------------------------- pick the best
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_kind = {"Random Forest": "rf", "XGBoost": "xgb"}.get(best_name)
    print("\n" + "=" * 78)
    print(f"BEST CLASSICAL BASELINE: {best_name}  "
          f"(macro-F1 {results[best_name]['macro_f1']:.4f}, "
          f"exact-match {results[best_name]['exact_match']:.4f})")
    print("=" * 78)
    out["best_baseline"] = dict(name=best_name, **{k: v for k, v in results[best_name].items()
                                                   if k != "per_label"})

    best_model = tuned.get(best_kind)
    Pte = preds[best_name]

    # ------------------------------------------------------- 4. importances
    print("\n" + "-" * 78)
    print("4. Feature importance (permutation, on validation)")
    print("-" * 78)
    imp = A.permutation_importances(best_model, Xva, Yva)
    mat = A.importance_matrix(imp)
    overlap = A.feature_overlap(mat)
    imp.to_csv(C.RESULTS / "feature_importance.csv", index=False)
    mat.to_csv(C.RESULTS / "feature_importance_matrix.csv")
    for a, feats in A.top_features(imp).items():
        print(f"    {a:15s} " + ", ".join(f"{f} ({v:.3f})" for f, v in feats))
    bh_sc = float(overlap.loc["beam_hardening", "scatter"])
    print(f"\n    shared-evidence similarity, beam_hardening vs scatter: {bh_sc:.3f}")
    others = [(f"{i}/{j}", float(overlap.loc[i, j])) for i in C.ARTIFACTS for j in C.ARTIFACTS
              if i < j and {i, j} != {"beam_hardening", "scatter"}]
    print("    other pairs: " + ", ".join(f"{k}={v:.2f}" for k, v in others))
    out["feature_overlap"] = overlap.round(4).to_dict()
    out["top_features"] = A.top_features(imp)

    # ------------------------------------------------------- 4b. shortcut ablation
    print("\n" + "-" * 78)
    print("4b. Shortcut ablation - is the score shape, or just brightness?")
    print("-" * 78)
    idx_all = {f: i for i, f in enumerate(C.FEATURE_NAMES)}
    subsets = {"all 14 features": C.FEATURE_NAMES,
               "shape only (no obj_mean/bg_level)": C.SHAPE_FEATURES,
               "level only (obj_mean, bg_level)": C.LEVEL_FEATURES}
    abl = {}
    for name, feats in subsets.items():
        cols = [idx_all[f] for f in feats]
        t = TunedMultiLabel(best_kind).fit(Xtr[:, cols], Ytr, Xva[:, cols], Yva,
                                           verbose=False)
        mm = M.multilabel_metrics(Yte, t.predict(Xte[:, cols]),
                                  t.predict_proba(Xte[:, cols]))
        abl[name] = mm
        per = "  ".join(f"{a[:4]}={mm['per_label'][a]['f1']:.3f}" for a in C.ARTIFACTS)
        print(f"    {name:36s} macro-F1={mm['macro_f1']:.4f}  "
              f"exact={mm['exact_match']:.4f}   {per}")
        # does removing the brightness shortcut make the look-alike pair confuse?
        cx = M.cross_label_confusion(Yte, t.predict(Xte[:, cols]))
        pair = cx[(cx.present == "scatter") & (cx.falsely_predicted == "beam_hardening")]
        rev = cx[(cx.present == "beam_hardening") & (cx.falsely_predicted == "scatter")]
        if len(pair) and len(rev):
            mm["false_bh_given_scatter"] = float(pair.false_rate.iloc[0])
            mm["false_scatter_given_bh"] = float(rev.false_rate.iloc[0])
            print(f"      {'':34s} BH/scatter confusion: "
                  f"scatter->BH {pair.false_rate.iloc[0]:.1%}, "
                  f"BH->scatter {rev.false_rate.iloc[0]:.1%}")
    out["shortcut_ablation"] = {k: {kk: vv for kk, vv in v.items()} for k, v in abl.items()}

    # ------------------------------------------------------- 5. difficulty
    print("\n" + "-" * 78)
    print("5. Difficulty structure (sets up experiments 3 and 4)")
    print("-" * 78)
    by_n = M.by_n_artifacts(Yte, Pte)
    per_combo = M.per_combination(Yte, Pte)
    cross = M.cross_label_confusion(Yte, Pte)
    by_n.to_csv(C.RESULTS / "accuracy_by_n_artifacts.csv", index=False)
    per_combo.to_csv(C.RESULTS / "accuracy_per_combination.csv", index=False)
    cross.to_csv(C.RESULTS / "cross_label_confusion.csv", index=False)
    print("    exact-match by number of artifacts:")
    for _, r in by_n.iterrows():
        print(f"      {int(r.n_artifacts)} artifacts  {r.exact_match:.3f}  (n={int(r.n_samples)})")
    print("    highest cross-label false-positive rates:")
    for _, r in cross.head(4).iterrows():
        print(f"      {r.present} present, {r.falsely_predicted} absent -> "
              f"falsely predicts {r.falsely_predicted} {r.false_rate:.1%}  (n={int(r.n)})")
    out["by_n_artifacts"] = by_n.to_dict("records")
    out["cross_label_confusion"] = cross.to_dict("records")

    # ------------------------------------------------------- 6. OOD severity
    if "test_ood_severity" in splits:
        print("\n" + "-" * 78)
        print("6. Held-out severity bands (experiment 5 reference point)")
        print("-" * 78)
        Xoo, Yoo = splits["test_ood_severity"]["X"], splits["test_ood_severity"]["Y"]
        moo = M.multilabel_metrics(Yoo, best_model.predict(Xoo), best_model.predict_proba(Xoo))
        results[f"{best_name} (OOD severity)"] = moo
        out["ood_severity"] = moo
        print(f"    in-band test     macro-F1 {results[best_name]['macro_f1']:.4f}  "
              f"exact-match {results[best_name]['exact_match']:.4f}")
        print(f"    OOD severity     macro-F1 {moo['macro_f1']:.4f}  "
              f"exact-match {moo['exact_match']:.4f}")

    # ------------------------------------------------------- figures
    print("\n" + "-" * 78)
    print("7. Figures")
    print("-" * 78)
    P.feature_importance_heatmap(mat, overlap, C.RESULTS / "fig_feature_importance.png")
    P.confusion_grid(M.label_confusions(Yte, Pte), C.RESULTS / "fig_confusion_matrices.png",
                     f"Per-label confusion matrices - {best_name}, test split")
    P.model_comparison({k: v for k, v in results.items() if "OOD" not in k},
                       C.RESULTS / "fig_model_comparison.png")
    P.difficulty_plots(by_n, per_combo, C.RESULTS / "fig_difficulty.png")
    for f in ["fig_feature_importance.png", "fig_confusion_matrices.png",
              "fig_model_comparison.png", "fig_difficulty.png"]:
        print(f"    results/{f}")

    # ------------------------------------------------------- write-ups
    (C.RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))

    md = ["# Stage A - Classical baseline results", "",
          f"Dataset: `{root}` ({layout} layout)  ",
          f"Train {len(Ytr)} / val {len(Yva)} / test {len(Yte)} samples, phantom-disjoint.", "",
          "Protocol: hyperparameters selected on validation by AUROC, decision",
          "thresholds tuned on validation by F1, test evaluated once at the end.", ""]
    for name in results:
        md.append(M.to_markdown_table(results[name], name))
    md += ["### Shortcut ablation", "",
           "| feature set | macro-F1 | exact-match | " +
           " | ".join(a for a in C.ARTIFACTS) + " |",
           "|---|---|---|" + "---|" * len(C.ARTIFACTS)]
    for name, mm in abl.items():
        md.append(f"| {name} | {mm['macro_f1']:.4f} | {mm['exact_match']:.4f} | "
                  + " | ".join(f"{mm['per_label'][a]['f1']:.3f}" for a in C.ARTIFACTS) + " |")
    md.append("")
    md += ["### Rule-based thresholds chosen (on validation)", "",
           "| artifact | rule | val F1 |", "|---|---|---|"]
    for a, d in rules.summary().items():
        md.append(f"| {a} | `{d['rule']}` | {d['val_f1']:.4f} |")
    md += ["", "### Shared-evidence similarity between labels", "",
           "Cosine similarity of permutation-importance profiles. High = the two",
           "labels are read from the same features.", "",
           overlap.round(3).to_markdown(), "",
           "### Exact-match accuracy vs number of simultaneous artifacts", "",
           by_n.round(4).to_markdown(index=False), "",
           "### Cross-label false-positive rates", "",
           cross.round(4).to_markdown(index=False), ""]
    (C.RESULTS / "summary.md").write_text("\n".join(md))

    try:
        import joblib
        (C.RESULTS / "models").mkdir(exist_ok=True)
        for kind, t in tuned.items():
            joblib.dump(t, C.RESULTS / "models" / f"{kind}_multilabel.joblib")
        joblib.dump(rules, C.RESULTS / "models" / "rule_based.joblib")
        print("    results/models/*.joblib")
    except Exception as exc:
        print(f"    (model saving skipped: {exc})")

    print(f"\n    results/summary.md   <- paste-ready tables for the report")
    print(f"    results/metrics.json")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
