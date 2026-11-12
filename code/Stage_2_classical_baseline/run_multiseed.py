"""
Multi-seed repeats of the classical baselines.

The CNN-vs-classical gap is a headline claim, so the classical side needs error
bars too.  Tree ensembles are stochastic (bootstrap sampling, feature
subsampling, column subsampling), so re-fitting under different seeds gives a
genuine spread.

Hyperparameters are NOT re-searched per seed -- the configuration selected on
validation in run_baseline.py is kept fixed, and only the fitting randomness
changes.  Decision thresholds are re-tuned per seed on validation, because they
are part of the fitted model.

    python3 run_multiseed.py                 # ~3 min
    python3 run_multiseed.py --seeds 1 2 3 4 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import config as C
from data import load
import metrics as M
from models import _make, tune_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    t0 = time.time()

    sel = json.loads((C.RESULTS / "metrics.json").read_text())
    _, splits, root, layout = load()
    Xtr, Ytr = splits["train"]["X"], splits["train"]["Y"]
    Xva, Yva = splits["val"]["X"], splits["val"]["Y"]

    rows = []
    for kind, name in (("rf", "Random Forest"), ("xgb", "XGBoost")):
        if kind not in sel:
            continue
        params = {a: sel[kind]["selection"][a]["params"] for a in C.ARTIFACTS}
        print(f"\n{name} - hyperparameters held fixed, re-fitting under "
              f"{len(args.seeds)} seeds")
        for seed in args.seeds:
            preds, probs = {}, {}
            for i, a in enumerate(C.ARTIFACTS):
                p = dict(params[a]); p.pop("random_state", None)
                m = _make(kind, p)
                m.set_params(random_state=seed)
                m.fit(Xtr, Ytr[:, i])
                pv = m.predict_proba(Xva)[:, 1]
                thr, _ = tune_threshold(Yva[:, i], pv)
                preds[a], probs[a] = {}, {}
                for split in ("test", "test_ood_severity"):
                    pr = m.predict_proba(splits[split]["X"])[:, 1]
                    probs[a][split] = pr
                    preds[a][split] = (pr >= thr).astype(int)
            for split in ("test", "test_ood_severity"):
                Y = splits[split]["Y"]
                P = np.stack([preds[a][split] for a in C.ARTIFACTS], 1)
                S = np.stack([probs[a][split] for a in C.ARTIFACTS], 1)
                rep = M.multilabel_metrics(Y, P, S)
                cx = M.cross_label_confusion(Y, P)
                pair = {}
                for _, r in cx.iterrows():
                    if r["present"] == "scatter" and r["falsely_predicted"] == "beam_hardening":
                        pair["scatter_to_bh"] = r["false_rate"]
                    if r["present"] == "beam_hardening" and r["falsely_predicted"] == "scatter":
                        pair["bh_to_scatter"] = r["false_rate"]
                rows.append(dict(model=name, seed=seed, split=split,
                                 macro_f1=rep["macro_f1"],
                                 exact_match=rep["exact_match"],
                                 hamming=rep["hamming_loss"], **pair))
            t = [r for r in rows if r["model"] == name and r["seed"] == seed
                 and r["split"] == "test"][0]
            print(f"  seed {seed}: macro-F1={t['macro_f1']:.4f}  "
                  f"exact={t['exact_match']:.4f}")

    df = pd.DataFrame(rows)
    out = C.RESULTS / "multiseed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)

    print("\n" + "=" * 70)
    print("CLASSICAL BASELINES  (mean +/- sd over seeds)")
    print("=" * 70)
    md = ["# Stage A - multi-seed robustness", "",
          f"Seeds: {args.seeds}. Hyperparameters fixed at the validation-selected "
          "configuration; only fitting randomness and threshold tuning vary.", "",
          "| model | split | macro-F1 | exact-match |", "|---|---|---|---|"]
    for (model, split), g in df.groupby(["model", "split"]):
        def ms(col):
            return (f"{g[col].mean():.4f} ± {g[col].std(ddof=1):.4f}" if len(g) > 1
                    else f"{g[col].mean():.4f}")
        print(f"  {model:16s} {split:20s} macro-F1 {ms('macro_f1'):22s} "
              f"exact {ms('exact_match')}")
        md.append(f"| {model} | {split} | {ms('macro_f1')} | {ms('exact_match')} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out}/runs.csv and summary.md")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
