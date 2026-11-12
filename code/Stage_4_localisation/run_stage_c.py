"""
Stage C - Grad-CAM localisation for Project 29 (spec item 5).

Produces per-artifact heat-maps for the trained CNN AND scores them against the
dataset's ground-truth ring and metal masks, so the explainability claim is
measured rather than asserted.

    python3 run_stage_c.py                       # default checkpoint, layer4 + layer3
    python3 run_stage_c.py --layers layer3
    python3 run_stage_c.py --checkpoint resnet18_fixed_pretrained.pt
    python3 run_stage_c.py --split test_ood_severity

Needs Stage B to have been run (it loads a checkpoint and Stage B's dataset code).
"""
import argparse
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

# Stage B is imported as a library: its folder goes on sys.path so that its own
# `import config` resolves to ITS config.  Stage C deliberately has no config.py.
sys.path.insert(0, str(S.STAGE_B))
sys.path.insert(0, str(S.STAGE_B / "src"))
import config as BC                      # noqa: E402  (Stage B's config)
from dataset import make_splits          # noqa: E402
from model import MultiLabelResNet, pick_device   # noqa: E402

from gradcam import cams_for_all_labels  # noqa: E402
import localisation_metrics as LM        # noqa: E402
from occlusion import occlusion_test     # noqa: E402
import viz as V                          # noqa: E402


def load_masks(df, n_pix=256):
    """Ground-truth localisation masks per sample: {'ring': ..., 'metal': ...}."""
    out = []
    for _, r in df.iterrows():
        d = {}
        mp = str(r.get("mask_path") or "")
        if mp:
            base = Path(r["abs_path"]).parent.parent
            cand = base / mp if (base / mp).exists() else None
            if cand is None:
                alt = Path(r["abs_path"]).parent.parent.parent / mp
                cand = alt if alt.exists() else None
            if cand is not None:
                with np.load(cand) as z:
                    for k in z.files:
                        d[k] = z[k].astype(bool)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=S.CHECKPOINT)
    ap.add_argument("--layers", nargs="+", default=list(S.TARGET_LAYERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--dilate", type=int, default=S.MASK_DILATE_PX)
    args = ap.parse_args()

    t0 = time.time()
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("STAGE C - GRAD-CAM LOCALISATION")
    print("=" * 78)

    ckpt_path = S.STAGE_B / "results" / "models" / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}\nRun Stage B first.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    regime, init = ckpt["regime"], ckpt["init"]
    print(f"\ncheckpoint : {args.checkpoint}")
    print(f"regime     : {regime}   init: {init}")
    print(f"split      : {args.split}   mask dilation: {args.dilate} px")

    device = pick_device()
    model = MultiLabelResNet(init="scratch", severity_head=ckpt.get("severity_head", False))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    datasets, root, layout = make_splits(regime)
    ds = datasets[args.split]
    loader = DataLoader(ds, batch_size=S.BATCH_SIZE, shuffle=False)
    masks = load_masks(ds.df)
    n_with_mask = sum(1 for m in masks if m)
    print(f"samples    : {len(ds)}   with ground-truth masks: {n_with_mask}")

    out = dict(checkpoint=args.checkpoint, regime=regime, init=init,
               split=args.split, dilate_px=args.dilate, layers={})

    for layer in args.layers:
        print("\n" + "-" * 78)
        print(f"LAYER {layer}")
        print("-" * 78)
        cams, peaks, Y = cams_for_all_labels(model, loader, device, n_labels=4,
                                             layer_name=layer)
        print(f"  computed {cams.shape[0]} x {cams.shape[1]} maps "
              f"({cams.shape[2]}x{cams.shape[3]})")

        loc = {}
        for i, a in enumerate(BC.ARTIFACTS):
            if a not in S.LOCALISABLE:
                continue
            per_sample_mask = [m.get(a) for m in masks]
            sc = LM.score_localisable(cams[:, i], per_sample_mask, Y[:, i].astype(bool),
                                      dilate_px=args.dilate,
                                      thresh_frac=S.CAM_THRESHOLD)
            loc[a] = sc
            print(f"  {a:8s} n={sc['n']:3d}  mask covers {sc['mask_area_fraction']:.1%} "
                  f"of the image")
            print(f"           pointing={sc['pointing_game']:.3f} "
                  f"(chance {sc['pointing_chance']:.3f}, lift x{sc['pointing_lift']:.1f})  "
                  f"IoU={sc['iou']:.3f}  energy-ratio={sc['energy_ratio']:.2f}")

        conc = {}
        for i, a in enumerate(BC.ARTIFACTS):
            conc[a] = LM.score_concentration(cams[:, i], peaks[:, i], Y[:, i].astype(bool))
        print("  heat concentration (share in hottest 10% of pixels, present & non-empty):")
        for a in BC.ARTIFACTS:
            c = conc[a]
            tag = "localisable" if a in S.LOCALISABLE else "global"
            print(f"    {a:15s} {c.get('concentration_present', float('nan')):.3f} "
                  f"(n={c['n_scored']:3d})   empty maps: present "
                  f"{c['empty_fraction_present']:.0%}, absent "
                  f"{c['empty_fraction_absent']:.0%}   [{tag}]")

        out["layers"][layer] = dict(localisation=loc, concentration=conc)

        V.localisation_bars(loc, S.RESULTS / f"fig_localisation_{layer}.png")
        V.concentration_bars(conc, list(BC.ARTIFACTS),
                             S.RESULTS / f"fig_concentration_{layer}.png")

        # qualitative overlays: prefer multi-artifact samples that have masks
        # pick a SPREAD of label combinations, not just the all-four cases, so the
        # figure shows what the maps do when an artifact is absent as well
        pick, seen = [], set()
        for target in (0, 1, 1, 2, 2, 3, 4, 4):
            for j in np.argsort([abs(int(Y[k].sum()) - target) for k in range(len(Y))]):
                if j not in seen and int(Y[j].sum()) == target:
                    pick.append(int(j)); seen.add(int(j)); break
        for j in range(len(Y)):
            if len(pick) >= S.N_QUALITATIVE:
                break
            if j not in seen:
                pick.append(j); seen.add(j)
        pick = np.array(pick[:S.N_QUALITATIVE])
        imgs = np.stack([ds.x[j] for j in pick])
        V.overlay_grid(imgs, cams[pick], Y[pick], list(BC.ARTIFACTS),
                       [masks[j] for j in pick],
                       S.RESULTS / f"fig_gradcam_overlays_{layer}.png",
                       n=len(pick),
                       title=f"Grad-CAM per artifact - {regime}/{init}, {layer}  "
                             f"(white outline = ground-truth mask)")
        print(f"  figures: fig_localisation_{layer}.png, "
              f"fig_concentration_{layer}.png, fig_gradcam_overlays_{layer}.png")

    # ------------------------------------------------------- causal occlusion test
    print("\n" + "-" * 78)
    print("OCCLUSION TEST - is the labelled region what the decision depends on?")
    print("-" * 78)
    thr = np.array(ckpt["thresholds"])
    occ = {}
    for i, a in enumerate(BC.ARTIFACTS):
        if a not in S.LOCALISABLE:
            continue
        r = occlusion_test(model, ds.x, masks, ds.y.astype(int), i, a, thr, device)
        occ[a] = r
        if r.get("n"):
            print(f"  {a} (n={r['n']}):")
            for key, label in (("original", "original"),
                               ("region_erased", "region erased, rest kept"),
                               ("region_only", "only the region kept")):
                print(f"    {label:28s} mean P={r[key]['mean_prob']:.3f}  "
                      f"detected={r[key]['detected']:.1%}")
    out["occlusion"] = occ

    (S.RESULTS / "localisation_metrics.json").write_text(json.dumps(out, indent=2,
                                                                    default=float))

    md = ["# Stage C - Grad-CAM localisation", "",
          f"Model: `{args.checkpoint}` ({regime} / {init}). Split: {args.split}. ",
          f"Ring masks dilated by {args.dilate} px before overlap scoring.", ""]
    for layer, d in out["layers"].items():
        md += [f"## {layer}", "",
               "| artifact | n | mask area | pointing game | chance | lift | IoU | "
               "energy ratio |", "|---|---|---|---|---|---|---|---|"]
        for a, sc in d["localisation"].items():
            md.append(f"| {a} | {sc['n']} | {sc['mask_area_fraction']:.1%} | "
                      f"{sc['pointing_game']:.3f} | {sc['pointing_chance']:.3f} | "
                      f"x{sc['pointing_lift']:.1f} | {sc['iou']:.3f} | "
                      f"{sc['energy_ratio']:.2f} |")
        md += ["", "| artifact | kind | concentration (present) | empty maps, present | "
               "empty maps, absent |", "|---|---|---|---|---|"]
        for a in BC.ARTIFACTS:
            c = d["concentration"][a]
            kind = "localisable" if a in S.LOCALISABLE else "global"
            md.append(f"| {a} | {kind} | "
                      f"{c.get('concentration_present', float('nan')):.3f} | "
                      f"{c['empty_fraction_present']:.0%} | "
                      f"{c['empty_fraction_absent']:.0%} |")
        md.append("")
    md += ["## Causal occlusion test", "",
           "| artifact | n | original | region erased | only region kept |",
           "|---|---|---|---|---|"]
    for a, r in occ.items():
        if r.get("n"):
            md.append(f"| {a} | {r['n']} | "
                      f"{r['original']['mean_prob']:.3f} ({r['original']['detected']:.0%}) | "
                      f"{r['region_erased']['mean_prob']:.3f} "
                      f"({r['region_erased']['detected']:.0%}) | "
                      f"{r['region_only']['mean_prob']:.3f} "
                      f"({r['region_only']['detected']:.0%}) |")
    md.append("")
    (S.RESULTS / "summary.md").write_text("\n".join(md))

    print(f"\nwrote results/summary.md and localisation_metrics.json")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
