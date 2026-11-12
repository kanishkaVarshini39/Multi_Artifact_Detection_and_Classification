"""
Stage D - assemble the five required experiments (spec item 18) and produce the
evaluation deliverables (items 16, 17, 24).

Experiment 1 needs its own training runs: run `python3 run_experiment1.py` first
(about 25 minutes).  This script picks up its output if present and assembles
everything else from the trained Stage A and Stage B models.

    python3 run_stage_d.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import settings as S
import evaluate_shim as ES
import gather as G
import viz as V

M = ES.M
ARTIFACTS = M.C.ARTIFACTS


def pair_rates(cross_records):
    d = {}
    for r in cross_records:
        if r["present"] == "scatter" and r["falsely_predicted"] == "beam_hardening":
            d["scatter_to_bh"] = r["false_rate"]
        if r["present"] == "beam_hardening" and r["falsely_predicted"] == "scatter":
            d["bh_to_scatter"] = r["false_rate"]
    return d


def main():
    t0 = time.time()
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    md, out = [], {}
    print("=" * 78); print("STAGE D - REQUIRED EXPERIMENTS"); print("=" * 78)

    print("\ncollecting predictions ...")
    cls_name, cls = G.classical_predictions()
    ckpt_name, ck, cnn = G.cnn_predictions()
    cnn_name = f"CNN ({ck['regime']}/{ck['init']})"
    print(f"  classical : {cls_name}")
    print(f"  CNN       : {ckpt_name}")

    rep = {}
    for label, pack in ((cls_name, cls), (cnn_name, cnn)):
        rep[label] = {}
        for split in ("test", "test_ood_severity"):
            d = pack[split]
            rep[label][split] = M.multilabel_metrics(d["Y"], d["P"], d["prob"])

    md += ["# Stage D - required experiments (spec item 18)", "",
           f"Classical baseline: **{cls_name}**  ",
           f"CNN: **{ckpt_name}** ({ck['regime']} / {ck['init']})", ""]

    # ---------------------------------------------------------------- EXP 1
    print("\n" + "-" * 78)
    print("EXPERIMENT 1 - single-artifact vs combined training")
    print("-" * 78)
    e1_path = S.RESULTS / "experiment1.json"
    if e1_path.exists():
        e1 = json.loads(e1_path.read_text())
        out["experiment1"] = e1
        slices = ["single_only", "single_or_clean", "multi_artifact", "all"]
        md += ["## Experiment 1 - does combined training help or hurt single-artifact "
               "detection?", "",
               "Three arms. The brief's version compares only A and C, which confounds "
               "training COMPOSITION with training SIZE (C has 2.7x the data). Arm B is "
               "size-matched to A, so A-vs-B isolates composition and B-vs-C isolates "
               "quantity.", "",
               "| arm | n train | " + " | ".join(slices) + " |",
               "|---|---|" + "---|" * len(slices)]
        for name, e in e1.items():
            row = [name, str(e["n_train"])]
            row += [f"{e['slices'][s]['macro_f1']:.3f} / {e['slices'][s]['exact_match']:.3f}"
                    for s in slices]
            md.append("| " + " | ".join(row) + " |")
        md.append("")
        md.append("Cells are macro-F1 / exact-match.")
        a, b, c = (e1["A_singles_only"], e1["B_combined_matched"], e1["C_combined_full"])
        k = "single_only"
        comp = b["slices"][k]["macro_f1"] - a["slices"][k]["macro_f1"]
        quant = c["slices"][k]["macro_f1"] - b["slices"][k]["macro_f1"]
        md += ["", f"On single-artifact test samples: composition effect "
               f"**{comp:+.4f}** macro-F1 (A→B, size held equal), data-quantity effect "
               f"**{quant:+.4f}** (B→C).", ""]
        for name, e in e1.items():
            print(f"  {name:22s} n={e['n_train']:5d}  "
                  + "  ".join(f"{s}={e['slices'][s]['macro_f1']:.3f}" for s in slices))
        print(f"  composition effect {comp:+.4f}   data-quantity effect {quant:+.4f}")
        V.exp1_figure(e1, slices, S.RESULTS / "fig_exp1_ablation.png")
    else:
        print("  experiment1.json not found - run run_experiment1.py first (skipped)")
        md += ["## Experiment 1", "", "_Not run: execute `run_experiment1.py` first._", ""]

    # ---------------------------------------------------------------- EXP 2
    print("\n" + "-" * 78); print("EXPERIMENT 2 - CNN vs classical"); print("-" * 78)
    sa = json.loads((S.STAGE_A / "results" / "metrics.json").read_text())
    models = {}
    for key, nm in (("rule_based", "rule-based"), ("rf", "Random Forest"), ("xgb", "XGBoost")):
        if key in sa and "test" in sa[key]:
            models[nm] = sa[key]["test"]
    models[cnn_name] = rep[cnn_name]["test"]
    per_label = {nm: [m["per_label"][a]["f1"] for a in ARTIFACTS] for nm, m in models.items()}
    aggregates = {nm: [m["macro_f1"], m["exact_match"], m["hamming_loss"]]
                  for nm, m in models.items()}
    out["experiment2"] = models
    md += ["## Experiment 2 - CNN vs classical baseline", "",
           "| model | " + " | ".join(ARTIFACTS) + " | macro-F1 | exact-match | Hamming |",
           "|---|" + "---|" * (len(ARTIFACTS) + 3)]
    for nm, m in models.items():
        md.append("| " + nm + " | "
                  + " | ".join(f"{m['per_label'][a]['f1']:.3f}" for a in ARTIFACTS)
                  + f" | **{m['macro_f1']:.4f}** | **{m['exact_match']:.4f}** | "
                  f"{m['hamming_loss']:.4f} |")
    md.append("")
    for nm, m in models.items():
        print(f"  {nm:26s} macro-F1={m['macro_f1']:.4f}  exact={m['exact_match']:.4f}")
    best_cls = max((m for nm, m in models.items() if nm != cnn_name),
                   key=lambda m: m["macro_f1"])
    gain_f1 = models[cnn_name]["macro_f1"] - best_cls["macro_f1"]
    gain_em = models[cnn_name]["exact_match"] - best_cls["exact_match"]
    md += [f"CNN over the best classical baseline: **{gain_f1:+.4f}** macro-F1, "
           f"**{gain_em:+.4f}** exact-match.", ""]
    print(f"  -> CNN gain: {gain_f1:+.4f} macro-F1, {gain_em:+.4f} exact-match")
    V.exp2_figure(ARTIFACTS, per_label, aggregates, S.RESULTS / "fig_exp2_cnn_vs_classical.png")

    # ---------------------------------------------------------------- EXP 3
    print("\n" + "-" * 78); print("EXPERIMENT 3 - beam hardening vs scatter"); print("-" * 78)
    rates, cms = {}, {}
    for label, pack in ((cls_name, cls), (cnn_name, cnn)):
        d = pack["test"]
        rates[label] = pair_rates(M.cross_label_confusion(d["Y"], d["P"]).to_dict("records"))
        cms[label] = M.label_confusions(d["Y"], d["P"])
    if "shortcut_ablation" in sa:
        for nm, mm in sa["shortcut_ablation"].items():
            if "false_bh_given_scatter" in mm:
                rates[f"{cls_name} [{nm}]"] = dict(
                    scatter_to_bh=mm["false_bh_given_scatter"],
                    bh_to_scatter=mm["false_scatter_given_bh"])
    out["experiment3"] = rates
    md += ["## Experiment 3 - beam-hardening / scatter confusion", "",
           "| model | scatter present → false BH | BH present → false scatter |",
           "|---|---|---|"]
    for nm, r in rates.items():
        md.append(f"| {nm} | {r.get('scatter_to_bh', 0):.1%} | "
                  f"{r.get('bh_to_scatter', 0):.1%} |")
    md.append("")
    for nm, r in rates.items():
        print(f"  {nm:48s} scatter→BH {r.get('scatter_to_bh', 0):.1%}  "
              f"BH→scatter {r.get('bh_to_scatter', 0):.1%}")
    V.exp3_figure(rates, S.RESULTS / "fig_exp3_bh_scatter.png")
    V.confusion_matrices(cms, S.RESULTS / "fig_confusion_matrices.png",
                         "Per-label confusion matrices, test split (spec item 16)")

    # ---------------------------------------------------------------- EXP 4
    print("\n" + "-" * 78); print("EXPERIMENT 4 - vs number of artifacts"); print("-" * 78)
    by_count, combos = {}, {}
    for label, pack in ((cls_name, cls), (cnn_name, cnn)):
        d = pack["test"]
        bn = M.by_n_artifacts(d["Y"], d["P"])
        vals = [float(bn[bn.n_artifacts == k]["exact_match"].iloc[0])
                if (bn.n_artifacts == k).any() else np.nan for k in range(5)]
        by_count[label] = vals
        combos[label] = M.per_combination(d["Y"], d["P"])
        print(f"  {label:26s} " + "  ".join(f"{k}:{v:.3f}" for k, v in enumerate(vals)))
    out["experiment4"] = {k: v for k, v in by_count.items()}
    md += ["## Experiment 4 - accuracy vs number of simultaneous artifacts", "",
           "| model | 0 | 1 | 2 | 3 | 4 |", "|---|---|---|---|---|---|"]
    for nm, v in by_count.items():
        md.append(f"| {nm} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    md += ["", "### Per-combination accuracy, 16 classes (spec item 16)", "",
           "| combo | n | " + " | ".join(combos) + " |",
           "|---|---|" + "---|" * len(combos)]
    ref = combos[cnn_name]
    for _, r in ref.iterrows():
        row = [r["combo"], str(int(r["n_samples"]))]
        for nm in combos:
            m = combos[nm]
            v = m[m.combo == r["combo"]]["exact_match"]
            row.append(f"{float(v.iloc[0]):.3f}" if len(v) else "-")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    ref.to_csv(S.RESULTS / "per_combination_cnn.csv", index=False)
    combos[cls_name].to_csv(S.RESULTS / "per_combination_classical.csv", index=False)
    V.exp4_figure(by_count, combos, S.RESULTS / "fig_exp4_by_count.png")

    # ---------------------------------------------------------------- EXP 5
    print("\n" + "-" * 78); print("EXPERIMENT 5 - held-out severities"); print("-" * 78)
    series = {}
    for label in (cls_name, cnn_name):
        series[label] = dict(
            macro_f1=[rep[label]["test"]["macro_f1"], rep[label]["test_ood_severity"]["macro_f1"]],
            exact_match=[rep[label]["test"]["exact_match"],
                         rep[label]["test_ood_severity"]["exact_match"]])
        print(f"  {label:26s} macro-F1 {series[label]['macro_f1'][0]:.4f} -> "
              f"{series[label]['macro_f1'][1]:.4f}   exact "
              f"{series[label]['exact_match'][0]:.4f} -> {series[label]['exact_match'][1]:.4f}")
    out["experiment5"] = series
    md += ["## Experiment 5 - generalisation to held-out severity bands", "",
           "| model | macro-F1 in-band | macro-F1 held-out | exact in-band | "
           "exact held-out |", "|---|---|---|---|---|"]
    for nm, s in series.items():
        md.append(f"| {nm} | {s['macro_f1'][0]:.4f} | {s['macro_f1'][1]:.4f} | "
                  f"{s['exact_match'][0]:.4f} | {s['exact_match'][1]:.4f} |")
    md.append("")
    V.exp5_figure(series, S.RESULTS / "fig_exp5_severity.png")

    # ---------------------------------------------------------------- write
    for label in rep:
        md.append(M.to_markdown_table(rep[label]["test"], f"{label} - test split"))
        md.append(M.to_markdown_table(rep[label]["test_ood_severity"],
                                      f"{label} - held-out severity"))
    (S.RESULTS / "summary.md").write_text("\n".join(md))
    (S.RESULTS / "all_experiments.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote results/summary.md, all_experiments.json and 6 figures")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
