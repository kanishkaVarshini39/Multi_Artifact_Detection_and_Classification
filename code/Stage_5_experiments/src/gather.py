"""
Collect predictions from the trained models so every experiment is built from
actual predictions rather than from parsing other stages' summary files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as S


def classical_predictions():
    """
    Read the best Stage A baseline's exported predictions.

    Deliberately NOT by unpickling Stage A's model: that requires Stage A's
    modules on sys.path, which would shadow Stage B's `config` for everything
    loaded afterwards.  Stage A exports plain arrays instead
    (`python3 export_predictions.py` in that folder).
    """
    f = S.STAGE_A / "results" / "predictions.npz"
    if not f.exists():
        raise SystemExit(
            f"{f} not found.\nRun:  cd '{S.STAGE_A}' && python3 export_predictions.py")
    z = np.load(f, allow_pickle=False)
    name = str(z["model_name"])
    out = {}
    for split in ("test", "test_ood_severity"):
        if f"{split}__Y" not in z:
            continue
        out[split] = dict(Y=z[f"{split}__Y"], P=z[f"{split}__pred"],
                          prob=z[f"{split}__prob"],
                          sample_id=z[f"{split}__sample_id"])
    return name, out


def cnn_predictions(checkpoint=None):
    """The chosen Stage B CNN, re-run on test and OOD."""
    sys.path.insert(0, str(S.STAGE_B)); sys.path.insert(0, str(S.STAGE_B / "src"))
    from dataset import make_splits
    from model import MultiLabelResNet, pick_device
    from train import predict

    ckpt_name = checkpoint or S.REFERENCE_CHECKPOINT
    ck = torch.load(S.STAGE_B / "results" / "models" / ckpt_name, map_location="cpu")
    model = MultiLabelResNet(init="scratch", severity_head=ck.get("severity_head", False))
    model.load_state_dict(ck["state_dict"])
    device = pick_device(); model.to(device).eval()
    thr = np.array(ck["thresholds"])

    datasets, _, _ = make_splits(ck["regime"])
    out = {}
    for split in ("test", "test_ood_severity"):
        P, Y, _ = predict(model, datasets[split], device)
        out[split] = dict(Y=Y, P=(P >= thr[None, :]).astype(int), prob=P,
                          meta=datasets[split].df)
    return ckpt_name, ck, out
