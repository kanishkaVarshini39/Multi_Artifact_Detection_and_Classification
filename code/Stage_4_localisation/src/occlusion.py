"""
Causal occlusion test.

Grad-CAM says where a network *appears* to look.  It cannot say whether that
region is what the decision actually depends on, and it is unreliable on a
saturated output.  Occlusion answers the causal question directly: remove a
region, re-run the model, and see whether the prediction survives.

Two complementary edits per artifact that has a ground-truth mask:

  erase region    blank the labelled region, keep everything else.
                  Prediction survives  -> the labelled region was NOT the evidence.
  keep region     blank everything except the labelled region.
                  Prediction survives  -> the labelled region IS sufficient.

Blanking uses the local median of a surrounding annulus rather than zero, so the
edit does not itself introduce a high-contrast edge that the network could read
as a new artifact.
"""
import numpy as np
import torch

from localisation_metrics import dilate


def _fill_region(img, region, surround_px=12, pad=3):
    out = img.copy()
    inner = dilate(region, pad)
    ring = dilate(region, surround_px) & ~inner
    fill = float(np.median(img[ring])) if ring.any() else float(np.median(img))
    out[inner] = fill
    return out


def _keep_region(img, region, pad=3):
    out = img.copy()
    inner = dilate(region, pad)
    out[~inner] = float(np.median(img[~inner]))
    return out


@torch.no_grad()
def _probs(model, arr, device, batch_size=32):
    out = []
    for k in range(0, len(arr), batch_size):
        t = torch.from_numpy(arr[k:k + batch_size]).unsqueeze(1).to(device)
        out.append(torch.sigmoid(model(t)[0]).cpu().numpy())
    return np.concatenate(out)


def occlusion_test(model, images, masks, labels, artifact_index, artifact_name,
                   thresholds, device):
    """Returns the mean probability and detection rate under each edit."""
    sel = [j for j, (m, y) in enumerate(zip(masks, labels))
           if y[artifact_index] == 1 and artifact_name in m and m[artifact_name].any()]
    if not sel:
        return dict(n=0)

    X = images[sel].copy()
    Xe = np.stack([_fill_region(X[i], masks[j][artifact_name])
                   for i, j in enumerate(sel)])
    Xk = np.stack([_keep_region(X[i], masks[j][artifact_name])
                   for i, j in enumerate(sel)])

    thr = thresholds[artifact_index]
    out = dict(n=len(sel))
    for key, arr in (("original", X), ("region_erased", Xe), ("region_only", Xk)):
        p = _probs(model, arr, device)[:, artifact_index]
        out[key] = dict(mean_prob=float(p.mean()),
                        detected=float(np.mean(p >= thr)))
    return out
