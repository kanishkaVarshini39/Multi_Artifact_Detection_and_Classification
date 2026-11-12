"""
Grad-CAM for a multi-label network.

Standard formulation (Selvaraju et al.): hook a convolutional block, take the
gradient of the target logit with respect to its activations, average those
gradients spatially to get one weight per channel, combine the channels with
those weights, and keep the positive part.

Implemented directly rather than via pytorch-grad-cam: it is about thirty lines,
needs no extra dependency, runs on the MPS backend, and -- the real reason --
multi-label targeting has to be explicit.  There is no single "predicted class"
here; each of the four outputs gets its own map.

Batched: gradients do not mix between samples, so one backward pass on the SUM
of a target logit across the batch yields every sample's map at once.  Four
passes over the data produce all four maps.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer_name="layer4"):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        layer = dict(model.backbone.named_modules())[layer_name]
        self._h1 = layer.register_forward_hook(self._save_activation)
        self._h2 = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x, target_index, out_size=None):
        """
        x: (B, 1, H, W) on the model's device
        target_index: which of the four outputs to explain
        Returns ((B, H, W) maps normalised to [0, 1], (B,) raw maxima).
        """
        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(x)
        logits[:, target_index].sum().backward()

        A = self.activations                       # (B, C, h, w)
        G = self.gradients                         # (B, C, h, w)
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))

        size = out_size or x.shape[-2:]
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).detach().cpu().numpy()

        flat = cam.reshape(len(cam), -1)
        mx = flat.max(axis=1)
        # A confidently-NEGATIVE logit gives gradients that ReLU removes entirely,
        # so the map comes back exactly all-zero.  Normalising that by its own
        # maximum manufactures structure out of nothing, so the raw maximum is
        # returned too and empty maps are excluded downstream.
        return cam / np.maximum(mx[:, None, None], 1e-12), mx


def cams_for_all_labels(model, loader, device, n_labels=4, layer_name="layer4",
                        out_size=(256, 256)):
    """Returns (maps, raw_peaks, labels) in loader order."""
    cam = GradCAM(model, layer_name)
    per_label, per_peak, labels = [], [], None
    try:
        for t in range(n_labels):
            maps, peaks, ys = [], [], []
            for x, y, _ in loader:
                m, mx = cam(x.to(device), t, out_size=out_size)
                maps.append(m); peaks.append(mx); ys.append(y.numpy())
            per_label.append(np.concatenate(maps))
            per_peak.append(np.concatenate(peaks))
            if labels is None:
                labels = np.concatenate(ys).astype(int)
    finally:
        cam.remove()
    return np.stack(per_label, axis=1), np.stack(per_peak, axis=1), labels
