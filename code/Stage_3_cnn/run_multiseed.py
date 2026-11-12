"""
Multi-seed repeats of the Stage B 2x2 grid.

A preprint reporting point estimates from a single run invites the obvious
objection: how much of that difference is seed noise?  This re-runs every arm
under several seeds and reports mean +/- standard deviation.

It also reports PAIRED differences -- computed within each seed, then averaged.
Pairing is the stronger analysis here because the two arms of a comparison share
the same data order and initialisation stream, so seed-to-seed variation that
affects both cancels out.  Two claims in particular need this:

  regime  (fixed - per_image)      "removing the brightness cue costs nothing"
  init    (pretrained - scratch)   resolves specification defect D5

Results are written after EVERY run, so a long sweep that is interrupted still
leaves usable partial output.

    python3 run_multiseed.py                      # 3 seeds x 4 arms, ~2.7 h
    python3 run_multiseed.py --seeds 11 22 33
    python3 run_multiseed.py --inits pretrained   # the fast half, ~45 min
"""
import argparse
import gc
import json
import time
from pathlib import Path
import sys

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

OUT = C.RESULTS / "multiseed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20261017, 20261018, 20261019])
    ap.add_argument("--regimes", nargs="+", default=list(C.REGIMES))
    ap.add_argument("--inits", nargs="+", default=list(C.INITS))
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--summarise-only", action="store_true",
                    help="re-read saved runs and rewrite the summaries, train nothing")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    store = OUT / "runs.json"
    runs = json.loads(store.read_text()) if store.exists() else {}
    device = pick_device()
    t0 = time.time()

    total = len(args.regimes) * len(args.seeds) * len(args.inits)
    print("=" * 78)
    print(f"MULTI-SEED REPEATS - {total} runs "
          f"({len(args.seeds)} seeds x {len(args.regimes)} regimes x {len(args.inits)} inits)")
    print("=" * 78)
    print(f"device {device.type}   seeds {args.seeds}")

    if args.summarise_only:
        print(f"summarising {len(runs)} saved run(s), training nothing")
        summarise(runs, args)
        return

    done = 0
    for regime in args.regimes:
        for seed in args.seeds:
            datasets, root, layout = make_splits(regime, seed=seed)
            for init in args.inits:
                key = f"{regime}|{init}|{seed}"
                done += 1
                if key in runs:
                    print(f"\n[{done}/{total}] {key}  (already done, skipping)")
                    continue
                print(f"\n[{done}/{total}] {key}")
                model, hist = train_model(datasets, init=init, epochs=args.epochs,
                                          seed=seed, verbose=False)

                Pv, Yv, _ = predict(model, datasets["val"], device)
                thr = E.tune_thresholds(Pv, Yv)
                rec = dict(regime=regime, init=init, seed=seed,
                           best_epoch=hist["best_epoch"], seconds=hist["seconds"],
                           thresholds=thr.tolist())
                for split in ("test", "test_ood_severity"):
                    P, Y, _ = predict(model, datasets[split], device)
                    rep, _ = E.full_report(P, Y, thr)
                    rec[split] = dict(
                        macro_f1=rep["macro_f1"], exact_match=rep["exact_match"],
                        hamming=rep["hamming_loss"],
                        per_label={a: rep["per_label"][a]["f1"] for a in C.ARTIFACTS},
                        **E.bh_scatter_pair(rep["cross_label_confusion"]))
                runs[key] = rec
                store.write_text(json.dumps(runs, indent=2, default=float))
                print(f"    test macro-F1={rec['test']['macro_f1']:.4f}  "
                      f"exact={rec['test']['exact_match']:.4f}  "
                      f"OOD macro-F1={rec['test_ood_severity']['macro_f1']:.4f}  "
                      f"({hist['seconds']:.0f}s, best epoch {hist['best_epoch']})")
                del model
                gc.collect()
            del datasets
            gc.collect()

    summarise(runs, args)
    print(f"\ntotal {time.time() - t0:.0f}s")


def summarise(runs, args):
    rows = []
    for key, r in runs.items():
        rows.append(dict(regime=r["regime"], init=r["init"], seed=r["seed"],
                         macro_f1=r["test"]["macro_f1"],
                         exact_match=r["test"]["exact_match"],
                         ood_macro_f1=r["test_ood_severity"]["macro_f1"],
                         ood_exact=r["test_ood_severity"]["exact_match"],
                         scatter_to_bh=r["test"].get("scatter_to_bh", np.nan),
                         bh_to_scatter=r["test"].get("bh_to_scatter", np.nan),
                         best_epoch=r["best_epoch"], seconds=r["seconds"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df.to_csv(OUT / "runs.csv", index=False)

    metrics = ["macro_f1", "exact_match", "ood_macro_f1", "ood_exact",
               "scatter_to_bh", "bh_to_scatter"]
    agg = df.groupby(["regime", "init"])[metrics].agg(["mean", "std", "count"])
    agg.to_csv(OUT / "summary_by_arm.csv")

    print("\n" + "=" * 78)
    print("PER-ARM SUMMARY  (mean +/- sd over seeds)")
    print("=" * 78)
    print(f"{'arm':26s} {'n':>2s} {'macro-F1':>18s} {'exact-match':>18s} {'OOD macro-F1':>18s}")
    for (regime, init), g in df.groupby(["regime", "init"]):
        name = f"{regime}/{init}"
        def ms(col):
            return f"{g[col].mean():.4f}+/-{g[col].std(ddof=1):.4f}" if len(g) > 1 \
                else f"{g[col].mean():.4f}   (n=1)"
        print(f"{name:26s} {len(g):2d} {ms('macro_f1'):>18s} "
              f"{ms('exact_match'):>18s} {ms('ood_macro_f1'):>18s}")

    # ---- paired differences, computed within each seed
    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCES  (computed within each seed, then averaged)")
    print("=" * 78)
    paired = {}

    def report(label, pivot_col, a, b, fixed_other, other_name):
        sub = df[df[other_name] == fixed_other]
        p = sub.pivot_table(index="seed", columns=pivot_col, values=metrics)
        if a not in p.columns.get_level_values(1) or b not in p.columns.get_level_values(1):
            return
        out = {}
        for m in ["macro_f1", "exact_match", "ood_macro_f1"]:
            d = (p[(m, a)] - p[(m, b)]).dropna()
            if len(d) == 0:
                continue
            out[m] = dict(mean=float(d.mean()),
                          sd=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                          n=int(len(d)), values=[float(x) for x in d])
            sd = f"+/-{out[m]['sd']:.4f}" if len(d) > 1 else "(n=1)"
            print(f"  {label:34s} {m:14s} {out[m]['mean']:+.4f} {sd}  n={len(d)}")
        paired[label] = out

    for init in sorted(df["init"].unique()):
        report(f"fixed - per_image  [{init}]", "regime", "fixed", "per_image", init, "init")
    for regime in sorted(df["regime"].unique()):
        report(f"pretrained - scratch  [{regime}]", "init", "pretrained", "scratch",
               regime, "regime")

    (OUT / "paired_differences.json").write_text(json.dumps(paired, indent=2, default=float))

    md = ["# Stage B - multi-seed robustness", "",
          f"Seeds: {sorted(df['seed'].unique().tolist())}. "
          f"Each arm run {int(df.groupby(['regime','init']).size().max())} time(s).", "",
          "## Per-arm results (mean +/- sd)", "",
          "| arm | n | macro-F1 | exact-match | OOD macro-F1 | OOD exact |",
          "|---|---|---|---|---|---|"]
    for (regime, init), g in df.groupby(["regime", "init"]):
        def ms(col):
            return (f"{g[col].mean():.4f} ± {g[col].std(ddof=1):.4f}" if len(g) > 1
                    else f"{g[col].mean():.4f}")
        md.append(f"| {regime}/{init} | {len(g)} | {ms('macro_f1')} | {ms('exact_match')} | "
                  f"{ms('ood_macro_f1')} | {ms('ood_exact')} |")
    md += ["", "## Paired differences (within seed)", "",
           "| comparison | metric | difference |", "|---|---|---|"]
    for label, out in paired.items():
        for m, d in out.items():
            sd = f" ± {d['sd']:.4f}" if np.isfinite(d.get("sd", np.nan)) else ""
            md.append(f"| {label} | {m} | {d['mean']:+.4f}{sd} (n={d['n']}) |")
    md.append("")
    (OUT / "summary.md").write_text("\n".join(md))
    print(f"\nwrote {OUT}/runs.csv, summary_by_arm.csv, "
          f"paired_differences.json, summary.md")


if __name__ == "__main__":
    main()
