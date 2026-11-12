"""Training loop with early stopping on validation macro-F1 (spec item 14)."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from model import MultiLabelResNet, count_parameters, pick_device


def seed_everything(seed=C.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


@torch.no_grad()
def predict(model, ds, device, batch_size=64):
    """Returns (probabilities, labels, severity_predictions_or_None)."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=C.NUM_WORKERS)
    P, Y, S = [], [], []
    for x, y, _ in dl:
        logits, sev = model(x.to(device))
        P.append(torch.sigmoid(logits).cpu().numpy())
        Y.append(y.numpy())
        if sev is not None:
            S.append(sev.cpu().numpy())
    return (np.concatenate(P), np.concatenate(Y).astype(int),
            np.concatenate(S) if S else None)


def train_model(datasets, init="pretrained", severity_head=False, epochs=C.EPOCHS,
                verbose=True, seed=C.SEED, tag=""):
    seed_everything(seed)
    device = pick_device()
    model = MultiLabelResNet(init=init, severity_head=severity_head).to(device)

    tr = DataLoader(datasets["train"], batch_size=C.BATCH_SIZE, shuffle=True,
                    num_workers=C.NUM_WORKERS, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=C.LR_STEP, gamma=C.LR_GAMMA)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best = dict(macro_f1=-1.0, epoch=-1, state=None)
    history, stale, t0 = [], 0, time.time()
    if verbose:
        print(f"    {tag}device={device.type}  params={count_parameters(model):,}  "
              f"init={init}  severity_head={severity_head}")

    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y, s in tr:
            x, y, s = x.to(device), y.to(device), s.to(device)
            opt.zero_grad(set_to_none=True)
            logits, sev = model(x)
            loss = bce(logits, y)
            if sev is not None:
                loss = loss + C.SEVERITY_LOSS_WEIGHT * mse(sev, s)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(x)
            n += len(x)
        sched.step()

        Pv, Yv, _ = predict(model, datasets["val"], device)
        f1 = f1_score(Yv, (Pv >= 0.5).astype(int), average="macro", zero_division=0)
        history.append(dict(epoch=epoch, train_loss=tot / max(n, 1), val_macro_f1=float(f1)))

        if f1 > best["macro_f1"] + 1e-5:
            best = dict(macro_f1=float(f1), epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"      epoch {epoch:3d}  loss={tot/max(n,1):.4f}  "
                  f"val macro-F1={f1:.4f}  best={best['macro_f1']:.4f} @{best['epoch']}")
        if stale >= C.EARLY_STOP_PATIENCE:
            if verbose:
                print(f"      early stop at epoch {epoch} "
                      f"(no improvement for {C.EARLY_STOP_PATIENCE})")
            break

    model.load_state_dict(best["state"])
    if verbose:
        print(f"      done in {time.time()-t0:.0f}s, best val macro-F1 "
              f"{best['macro_f1']:.4f} @ epoch {best['epoch']}")
    return model, dict(history=history, best_epoch=best["epoch"],
                       best_val_macro_f1=best["macro_f1"],
                       seconds=round(time.time() - t0, 1), device=device.type)
