"""
Experiment 1 - does combined-corruption training help or hurt single-artifact
detection?  (spec item 18, experiment 1)

Three arms, all using the Stage B winning configuration (per_image / pretrained)
so that training COMPOSITION is the only variable, plus a size control:

  A  single-artifact data only          630 samples
  B  combined data, size-matched        630 samples
  C  full combined dataset             1680 samples   (the Stage B model, retrained
                                                       here under the same seed)

Model SELECTION (early stopping) uses the full validation set for every arm, so
that criterion is identical and only the training data differs.

Decision THRESHOLDS are tuned on a validation subset MATCHED to each arm's own
training distribution.  This matters more than it sounds: tuning arm A on the
full validation set -- which is full of combinations it was never trained on --
drove all four thresholds to the floor (0.04), maximising recall and wrecking
its clean-image accuracy (18/45 correct).  Matched tuning gives arm A 45/45 on
clean images and 1.000 exact-match on single-artifact samples.  Both variants
are reported, because the discrepancy is itself a finding about multi-label
operating points.

    python3 run_experiment1.py                # ~25 min on MPS
    python3 run_experiment1.py --eval-only    # re-score saved checkpoints
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import settings as S

sys.path.insert(0, str(S.STAGE_B)); sys.path.insert(0, str(S.STAGE_B / "src"))
import config as BC                              # noqa: E402
from dataset import make_splits                  # noqa: E402
from train import train_model, predict           # noqa: E402
from model import pick_device                    # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
import evaluate_shim as ES                       # noqa: E402
from subsets import make_arms, test_masks        # noqa: E402


def val_subset_for(arm, datasets):
    """Validation rows drawn from the same distribution the arm was trained on."""
    import numpy as np
    from torch.utils.data import Subset
    if arm != "A_singles_only":
        return datasets["val"]
    idx = np.where(datasets["val"].df["n_artifacts"].to_numpy() <= 1)[0]
    return Subset(datasets["val"], idx.tolist())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true",
                    help="re-score saved checkpoints without retraining")
    args = ap.parse_args()

    t0 = time.time()
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 1 - single-artifact vs combined training")
    print("=" * 78)

    datasets, root, layout = make_splits(S.REGIME)
    arms = make_arms(datasets["train"], S.SEED)
    masks = test_masks(datasets["test"].df)
    print(f"\ndevice {pick_device().type}   regime {S.REGIME}   init {S.INIT}")
    for name, (_, desc) in arms.items():
        print(f"  {name:22s} {desc}")
    print("  test slices: " + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items()))

    device = pick_device()
    # a clean-only slice makes the threshold effect visible
    masks["clean_only"] = masks["single_or_clean"] & ~masks["single_only"]
    out = {}
    for name, (train_subset, desc) in arms.items():
        print(f"\n--- {name} ---")
        ckpt_file = S.RESULTS / f"exp1_{name}.pt"
        if args.eval_only and ckpt_file.exists():
            from model import MultiLabelResNet
            ck = torch.load(ckpt_file, map_location="cpu")
            model = MultiLabelResNet(init="scratch")
            model.load_state_dict(ck["state_dict"]); model.to(device).eval()
            hist = dict(best_epoch=ck.get("best_epoch", -1), seconds=ck.get("seconds", 0.0))
            print("      (re-scoring the saved checkpoint)")
        else:
            ds = dict(datasets)
            ds["train"] = train_subset
            model, hist = train_model(ds, init=S.INIT, epochs=BC.EPOCHS, seed=S.SEED)

        Pt, Yt, _ = predict(model, datasets["test"], device)
        entry = dict(description=desc, n_train=len(train_subset),
                     best_epoch=hist["best_epoch"], seconds=hist["seconds"],
                     slices={}, slices_full_val_thresholds={})

        for tuning, val_ds in (("matched", val_subset_for(name, datasets)),
                               ("full_val", datasets["val"])):
            Pv, Yv, _ = predict(model, val_ds, device)
            thr = ES.tune_thresholds(Pv, Yv)
            target = entry["slices"] if tuning == "matched" \
                else entry["slices_full_val_thresholds"]
            for slice_name, m in masks.items():
                rep, _ = ES.full_report(Pt[m], Yt[m], thr)
                target[slice_name] = rep
            if tuning == "matched":
                entry["thresholds"] = thr.tolist()
                print("      thresholds (matched val): "
                      + ", ".join(f"{a[:4]}={t:.2f}" for a, t in zip(BC.ARTIFACTS, thr)))
                for slice_name in masks:
                    rep = target[slice_name]
                    print(f"      {slice_name:16s} n={rep['n']:3d}  "
                          f"macro-F1={rep['macro_f1']:.4f}  exact={rep['exact_match']:.4f}")
            else:
                entry["thresholds_full_val"] = thr.tolist()
        out[name] = entry

        torch.save(dict(state_dict=model.state_dict(), regime=S.REGIME, init=S.INIT,
                        thresholds=entry["thresholds"], arm=name,
                        best_epoch=hist["best_epoch"], seconds=hist["seconds"]),
                   ckpt_file)
        del model

    (S.RESULTS / "experiment1.json").write_text(json.dumps(out, indent=2, default=float))

    print("\n" + "=" * 78)
    print("EXPERIMENT 1 SUMMARY  (macro-F1 / exact-match)")
    print("=" * 78)
    hdr = f"{'arm':22s} {'n_train':>8s}"
    for k in masks:
        hdr += f" {k:>22s}"
    print(hdr)
    for name, e in out.items():
        row = f"{name:22s} {e['n_train']:8d}"
        for k in masks:
            r = e["slices"][k]
            row += f" {r['macro_f1']:.3f}/{r['exact_match']:.3f}".rjust(23)
        print(row)

    a, b, c = out["A_singles_only"], out["B_combined_matched"], out["C_combined_full"]
    key = "single_only"
    print(f"\non SINGLE-ARTIFACT test samples (the question the brief asks):")
    print(f"  A singles-only      macro-F1 {a['slices'][key]['macro_f1']:.4f}")
    print(f"  B combined, matched macro-F1 {b['slices'][key]['macro_f1']:.4f}  "
          f"(vs A: {b['slices'][key]['macro_f1']-a['slices'][key]['macro_f1']:+.4f}  "
          f"<- composition effect, data size held equal)")
    print(f"  C combined, full    macro-F1 {c['slices'][key]['macro_f1']:.4f}  "
          f"(vs B: {c['slices'][key]['macro_f1']-b['slices'][key]['macro_f1']:+.4f}  "
          f"<- data-quantity effect)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
