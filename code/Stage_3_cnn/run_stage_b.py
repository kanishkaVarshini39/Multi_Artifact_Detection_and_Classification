"""
Stage B - Multi-label CNN for Project 29.

Runs a 2x2 grid:  normalisation regime {fixed, per_image}  x  init {pretrained, scratch}

  * regime resolves the brightness-shortcut question Stage A exposed
  * init   resolves specification defect D5 ("pretrained-from-scratch")

    python3 run_stage_b.py                 # full grid, 100 epochs
    python3 run_stage_b.py --quick         # 3 epochs, smoke test
    python3 run_stage_b.py --severity      # add auxiliary severity-regression heads
    python3 run_stage_b.py --regimes per_image --inits pretrained

Outputs land in results/.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import config as C
from dataset import make_splits
from train import train_model, predict
from model import pick_device
import evaluate as E
import plots as P


def load_baseline():
    f = C.STAGE_A / "results" / "metrics.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    b = d.get("best_baseline")
    if not b:
        return None
    return dict(name=b.get("name", "classical"), macro_f1=b["macro_f1"],
                exact_match=b["exact_match"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--quick", action="store_true", help="3 epochs, smoke test")
    ap.add_argument("--severity", action="store_true", help="add severity heads")
    ap.add_argument("--regimes", nargs="+", default=list(C.REGIMES))
    ap.add_argument("--inits", nargs="+", default=list(C.INITS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    epochs = 3 if args.quick else args.epochs

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 78)
    print("STAGE B - MULTI-LABEL CNN")
    print("=" * 78)
    baseline = load_baseline()
    if baseline:
        print(f"\nStage A target: {baseline['name']}  "
              f"macro-F1 {baseline['macro_f1']:.4f}  exact-match {baseline['exact_match']:.4f}")
    print(f"device: {pick_device().type}   epochs: {epochs}   "
          f"severity heads: {args.severity}")

    runs, rows = {}, []
    for regime in args.regimes:
        print("\n" + "=" * 78)
        print(f"REGIME: {regime}" +
              ("   (global brightness available)" if regime == "fixed"
               else "   (global brightness removed - the honest test)"))
        print("=" * 78)
        datasets, root, layout = make_splits(regime)
        print(f"  dataset: {root} ({layout})")
        print("  " + "  ".join(f"{k}={len(v)}" for k, v in datasets.items()))

        for init in args.inits:
            name = f"{regime} / {init}"
            print(f"\n  --- {name} ---")
            model, hist = train_model(datasets, init=init, severity_head=args.severity,
                                      epochs=epochs, tag="")

            device = pick_device()
            Pv, Yv, _ = predict(model, datasets["val"], device)
            thr = E.tune_thresholds(Pv, Yv)
            print(f"      thresholds: " +
                  ", ".join(f"{a[:4]}={t:.2f}" for a, t in zip(C.ARTIFACTS, thr)))

            Pt, Yt, St = predict(model, datasets["test"], device)
            rep, _ = E.full_report(Pt, Yt, thr)
            pair = E.bh_scatter_pair(rep["cross_label_confusion"])
            print(f"      TEST  macro-F1={rep['macro_f1']:.4f}  "
                  f"exact-match={rep['exact_match']:.4f}  "
                  f"hamming={rep['hamming_loss']:.4f}")
            print(f"            per-label F1: " +
                  "  ".join(f"{a[:4]}={rep['per_label'][a]['f1']:.3f}" for a in C.ARTIFACTS))
            print(f"            BH/scatter confusion: "
                  f"scatter->BH {pair.get('scatter_to_bh', 0):.1%}, "
                  f"BH->scatter {pair.get('bh_to_scatter', 0):.1%}")

            entry = dict(regime=regime, init=init, train=hist,
                         thresholds=thr.tolist(), test=rep, pair=pair)

            if "test_ood_severity" in datasets:
                Po, Yo, So = predict(model, datasets["test_ood_severity"], device)
                rep_o, _ = E.full_report(Po, Yo, thr)
                entry["ood_severity"] = rep_o
                print(f"      OOD   macro-F1={rep_o['macro_f1']:.4f}  "
                      f"exact-match={rep_o['exact_match']:.4f}")
                if St is not None and So is not None:
                    entry["severity_test"] = E.severity_errors(St, datasets["test"].sev, Yt)
                    print("            severity MAE: " + "  ".join(
                        f"{a[:4]}={v['mae']:.3f}" for a, v in entry["severity_test"].items()))

            runs[name] = entry
            rows.append(dict(name=name, regime=regime, init=init,
                             macro_f1=rep["macro_f1"], exact_match=rep["exact_match"],
                             hamming=rep["hamming_loss"],
                             ood_macro_f1=entry.get("ood_severity", {}).get("macro_f1"),
                             ood_exact=entry.get("ood_severity", {}).get("exact_match"),
                             **pair))

            mdir = C.RESULTS / "models"; mdir.mkdir(exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), regime=regime, init=init,
                            thresholds=thr.tolist(), severity_head=args.severity),
                       mdir / f"resnet18_{regime}_{init}{args.tag}.pt")
            del model
            gc.collect()

        del datasets
        gc.collect()

    # ---------------------------------------------------------------- outputs
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / f"arm_comparison{args.tag}.csv", index=False)
    (C.RESULTS / f"results{args.tag}.json").write_text(
        json.dumps(dict(baseline=baseline, runs=runs), indent=2, default=float))

    print("\n" + "=" * 78)
    print("SUMMARY (test split)")
    print("=" * 78)
    cols = ["name", "macro_f1", "exact_match", "ood_macro_f1", "scatter_to_bh", "bh_to_scatter"]
    print(df[[c for c in cols if c in df]].to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))
    if baseline:
        best = df.loc[df.macro_f1.idxmax()]
        print(f"\nclassical baseline : macro-F1 {baseline['macro_f1']:.4f}  "
              f"exact {baseline['exact_match']:.4f}")
        print(f"best CNN arm       : macro-F1 {best.macro_f1:.4f}  "
              f"exact {best.exact_match:.4f}   ({best['name']})")
        print(f"delta              : {best.macro_f1 - baseline['macro_f1']:+.4f} macro-F1, "
              f"{best.exact_match - baseline['exact_match']:+.4f} exact-match")

    P.training_curves(runs, C.RESULTS / f"fig_training_curves{args.tag}.png")
    P.arm_comparison(rows, baseline, C.RESULTS / f"fig_arm_comparison{args.tag}.png")
    P.confusion_pair(rows, C.RESULTS / f"fig_bh_scatter_confusion{args.tag}.png")

    md = ["# Stage B - CNN results", "",
          f"Device: {pick_device().type}. Epochs: {epochs}. "
          f"Severity heads: {args.severity}.", ""]
    if baseline:
        md += [f"Stage A classical target: **macro-F1 {baseline['macro_f1']:.4f}, "
               f"exact-match {baseline['exact_match']:.4f}** ({baseline['name']}).", ""]
    md += ["| arm | macro-F1 | exact-match | Hamming | OOD macro-F1 | "
           "scatter->BH | BH->scatter |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        ood = f"{r['ood_macro_f1']:.4f}" if r.get("ood_macro_f1") is not None else "-"
        md.append(f"| {r['name']} | {r['macro_f1']:.4f} | {r['exact_match']:.4f} | "
                  f"{r['hamming']:.4f} | {ood} | "
                  f"{r.get('scatter_to_bh', 0):.3f} | {r.get('bh_to_scatter', 0):.3f} |")
    md.append("")
    for name, r in runs.items():
        md.append(E.M.to_markdown_table(r["test"], f"{name} - test split"))
    (C.RESULTS / f"summary{args.tag}.md").write_text("\n".join(md))

    print(f"\nwrote results/, figures and summary{args.tag}.md")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
