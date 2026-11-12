"""
ResNet-18 multi-label classifier (spec item 12).

Resolves specification defect D5.  Item 12 says "pretrained-from-scratch",
which is self-contradictory, so both are provided and compared rather than
guessed:

  pretrained  ImageNet weights, with conv1's three RGB channels AVERAGED into
              one.  Averaging (not slicing) preserves the learned filter
              response for a grey input.
  scratch     random initialisation.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


class MultiLabelResNet(nn.Module):
    def __init__(self, init="pretrained", n_labels=4, severity_head=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if init == "pretrained" else None
        net = torchvision.models.resnet18(weights=weights)

        old = net.conv1
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            if init == "pretrained":
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        net.conv1 = new

        self.n_features = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.classifier = nn.Linear(self.n_features, n_labels)
        self.severity = nn.Linear(self.n_features, n_labels) if severity_head else None

    def forward(self, x):
        f = self.backbone(x)
        logits = self.classifier(f)
        sev = torch.sigmoid(self.severity(f)) if self.severity is not None else None
        return logits, sev

    def features(self, x):
        return self.backbone(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
