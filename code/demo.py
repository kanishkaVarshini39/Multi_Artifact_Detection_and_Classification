#!/usr/bin/env python3
"""
Live end-to-end demonstration, on one image, in about a minute.

Touches every stage of the project in order:

    1  generate   phantoms.py + pipeline.py   build an object, corrupt it, reconstruct
    2  measure    features.py                 the 14 handcrafted measurements
    3  classify   model.py + a checkpoint     what the trained network predicts
    4  explain    gradcam.py                  where it says it is looking
    5  test it    occlusion.py                erase that region and ask again

Step 5 is the point of the whole thing: the metal insert is deleted and the
prediction does not move.

    python3 demo.py              # pauses between steps - press Enter to advance
    python3 demo.py --no-pause   # straight through
    python3 demo.py --seed 11    # a different object
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CT = ROOT / "Stage_1_dataset_generation"
SB = ROOT / "Stage_3_cnn"
SC = ROOT / "Stage_4_localisation"

C_DIM, C_OK, C_HI, C_WARN, C_OFF = "\033[90m", "\033[96m", "\033[1;97m", "\033[93m", "\033[0m"


def rule(n=74):
    print(C_DIM + "─" * n + C_OFF)


def step(n, title, files):
    print()
    rule()
    print(f"{C_HI}STEP {n}{C_OFF}  {title}")
    print(f"{C_DIM}         {files}{C_OFF}")
    rule()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--seed", type=int, default=2, help="which phantom to build")
    ap.add_argument("--save", default=str(ROOT / "demo_out"))
    args = ap.parse_args()
    out = Path(args.save); out.mkdir(parents=True, exist_ok=True)

    def wait(msg="press Enter"):
        if not args.no_pause:
            try:
                input(f"\n{C_DIM}   [{msg}]{C_OFF}")
            except EOFError:
                pass

    print(f"\n{C_HI}MULTI-ARTIFACT CT — LIVE RUN{C_OFF}")
    print(f"{C_DIM}one image, every stage of the pipeline{C_OFF}")

    # ---- imports up front, so the slow part is visible and not mid-demo
    t0 = time.time()
    print(f"\n{C_DIM}loading torch, the simulator and the trained network …{C_OFF}")
    import matplotlib
    matplotlib.use("macosx")
    import matplotlib.pyplot as plt
    import torch

    sys.path.insert(0, str(CT)); sys.path.insert(0, str(CT / "src"))
    import config as C
    from phantoms import make_phantom
    from pipeline import build_projection_cache, simulate
    from features import compute, FEATURE_NAMES
    from preprocess import normalise_per_image

    sys.path.insert(0, str(SB)); sys.path.insert(0, str(SB / "src"))
    from model import MultiLabelResNet, pick_device
    sys.path.insert(0, str(SC / "src"))
    from gradcam import GradCAM
    from occlusion import _fill_region, _probs

    ck_path = SB / "results/models/resnet18_per_image_pretrained.pt"
    ck = torch.load(ck_path, map_location="cpu")
    net = MultiLabelResNet(init="scratch"); net.load_state_dict(ck["state_dict"])
    dev = pick_device(); net.to(dev).eval()
    thr = np.array(ck["thresholds"])
    print(f"{C_DIM}ready in {time.time()-t0:.1f}s  ·  device: {dev.type}  ·  "
          f"checkpoint: {ck_path.name}{C_OFF}")

    plt.ion()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    fig.canvas.manager.set_window_title("live run")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    fig.tight_layout(); fig.show(); plt.pause(0.1)

    wait("begin")

    # ───────────────────────────── 1. generate
    step(1, "GENERATE an image, with the label fixed first",
         "Stage_1_dataset_generation/src/phantoms.py  ·  pipeline.py")
    chosen = ["beam_hardening", "ring", "metal"]
    labels = {a: (a in chosen) for a in C.ARTIFACTS}
    sev = {a: 0.8 for a in C.ARTIFACTS}
    print(f"   we decide what to inject  →  {C_OK}{', '.join(chosen)}{C_OFF}")
    print(f"   {C_DIM}that decision IS the label. nothing is annotated afterwards.{C_OFF}")

    t = time.time()
    ph = make_phantom(args.seed, C.MASTER_SEED + 991 * args.seed)
    cache = build_projection_cache(ph)
    img, params, masks = simulate(ph, cache, labels, sev, np.random.default_rng(7))
    print(f"\n   phantom → forward project → corrupt → reconstruct   "
          f"{C_DIM}({time.time()-t:.2f}s){C_OFF}")
    print(f"   image: {img.shape}  attenuation range "
          f"[{img.min():.4f}, {img.max():.4f}] per pixel")

    x = normalise_per_image(img)
    axes[0].imshow(x, cmap="gray", vmin=0, vmax=1); axes[0].set_title("1  generated")
    fig.canvas.draw(); plt.pause(0.1)
    wait()

    # ───────────────────────────── 2. measure
    step(2, "MEASURE the 14 handcrafted features",
         "Stage_1_dataset_generation/src/features.py")
    f = compute(img)
    for k in FEATURE_NAMES:
        tag = f"  {C_WARN}← overall brightness{C_OFF}" if k in ("obj_mean", "bg_level") else ""
        print(f"   {k:20s} {f[k]:>10.4f}{tag}")
    print(f"\n   {C_DIM}the classical baseline sees only these 14 numbers.{C_OFF}")
    wait()

    # ───────────────────────────── 3. classify
    step(3, "CLASSIFY with the trained network",
         "Stage_3_cnn/src/model.py  ·  resnet18_per_image_pretrained.pt")
    t = time.time()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(dev))[0])[0]
    p = p.cpu().numpy()
    print(f"   {C_DIM}forward pass in {time.time()-t:.2f}s{C_OFF}\n")
    for i, a in enumerate(C.ARTIFACTS):
        hit = p[i] >= thr[i]
        truth = "present" if labels[a] else "absent "
        mark = f"{C_OK}DETECTED{C_OFF}" if hit else f"{C_DIM}not detected{C_OFF}"
        ok = "✓" if hit == labels[a] else "✗"
        print(f"   {a:16s} P = {p[i]:.3f}   thr {thr[i]:.2f}   {mark:28s} "
              f"truth: {truth} {ok}")
    print(f"\n   {C_OK}all four labels correct{C_OFF}" if
          all((p[i] >= thr[i]) == labels[a] for i, a in enumerate(C.ARTIFACTS))
          else f"\n   {C_WARN}not all labels correct{C_OFF}")
    wait()

    # ───────────────────────────── 4. explain
    step(4, "EXPLAIN — where does it say it is looking",
         "Stage_4_localisation/src/gradcam.py")
    mi = list(C.ARTIFACTS).index("metal")
    cam = GradCAM(net, "layer4")
    maps, peak = cam(torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(dev), mi)
    cam.remove()
    axes[1].imshow(x, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(maps[0], cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("2  Grad-CAM, metal")
    fig.canvas.draw(); plt.pause(0.1)
    print(f"   heat map for the metal output, peak response {peak[0]:.3f}")
    print(f"   {C_DIM}this says where the network APPEARS to look. it cannot say{C_OFF}")
    print(f"   {C_DIM}whether the decision depends on that region.{C_OFF}")
    wait()

    # ───────────────────────────── 5. the causal test
    step(5, "TEST IT — erase the metal and ask again",
         "Stage_4_localisation/src/occlusion.py")
    region = masks["metal"]
    erased = _fill_region(x, region)
    axes[2].imshow(erased, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("3  metal erased")
    ys, xs = np.nonzero(region)
    for ax in (axes[0], axes[2]):
        ax.add_patch(plt.Circle((xs.mean(), ys.mean()),
                                0.5 * max(np.ptp(ys), np.ptp(xs)) + 9,
                                fill=False, edgecolor="#F2A65A", lw=1.8, ls=(0, (4, 3))))
    axes[3].imshow(np.abs(erased - x), cmap="magma")
    axes[3].set_title("4  what we removed")
    fig.canvas.draw(); plt.pause(0.1)

    p_after = _probs(net, erased[None, ...], dev)[0, mi]
    print(f"   the insert is gone — {int(region.sum())} pixels replaced by the "
          f"local median\n")
    print(f"   before:  P(metal) = {C_HI}{p[mi]:.3f}{C_OFF}")
    print(f"   after:   P(metal) = {C_HI}{p_after:.3f}{C_OFF}   "
          f"{C_DIM}(change: {p_after - p[mi]:+.4f}){C_OFF}")
    print()
    if p_after >= thr[mi]:
        print(f"   {C_WARN}the metal is gone and the network still says metal.{C_OFF}")
        print(f"   {C_DIM}it was never reading the insert. it reads the streaks "
              f"across the whole slice.{C_OFF}")
        print(f"   {C_DIM}the mask we score explanations against marks the wrong "
              f"region.{C_OFF}")
    else:
        print(f"   {C_WARN}detection collapsed — the insert was necessary here.{C_OFF}")

    png = out / "live_run.png"
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    rule()
    print(f"{C_DIM}figure saved to {png}{C_OFF}")
    print(f"{C_DIM}total {time.time()-t0:.1f}s{C_OFF}\n")
    if not args.no_pause:
        wait("Enter to close")


if __name__ == "__main__":
    main()
