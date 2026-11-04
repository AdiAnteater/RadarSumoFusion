"""PointNet for per-point segmentation (Qi et al., CVPR 2017).

Input  (B, C, N) per-point features, any N (the set is pooled by max).
Output (B, K, N) class logits.

  per-point MLP  C -> 64 -> 64            local features (kept for the head)
                 64 -> 64 -> 128 -> 1024  then max over the N points = global feature
  head           [local 64 | global 1024] -> 512 -> 256 -> 128 -> K

Optional feature T-Net (the 64x64 alignment of the paper) with the
orthogonality regulariser. Off by default: the fused cloud is already in a
fixed world frame, so alignment buys little and costs time.

Normalisation: BatchNorm (the paper) or GroupNorm (--norm gn). With BatchNorm the
running statistics lag the fast-changing weights and eval-mode predictions
collapse, so train.py re-estimates them on training batches before every
evaluation and checkpoint (recalibrate_bn).

Padding: the data loader pads short frames by REPEATING real points, which
leaves the max-pool unchanged, so no mask is needed inside the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


NORM = {"type": "bn"}   # set by PointNetSeg before building layers


def _norm1d(c):
    if NORM["type"] == "gn":
        return nn.GroupNorm(min(8, c), c)
    return nn.BatchNorm1d(c)


def _mlp(dims, dropout=0.0, last_act=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=False))
        layers.append(_norm1d(dims[i + 1]))
        if last_act or i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
        if dropout and i == len(dims) - 2:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TNet(nn.Module):
    def __init__(self, k: int, width: float = 1.0):
        super().__init__()
        w = lambda c: max(8, int(c * width))  # noqa: E731
        self.k = k
        self.feat = _mlp([k, w(64), w(128), w(1024)])
        self.fc = nn.Sequential(
            nn.Linear(w(1024), w(512), bias=False), nn.LayerNorm(w(512)), nn.ReLU(inplace=True),
            nn.Linear(w(512), w(256), bias=False), nn.LayerNorm(w(256)), nn.ReLU(inplace=True),
            nn.Linear(w(256), k * k))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        g = self.feat(x).max(dim=2).values
        m = self.fc(g).view(-1, self.k, self.k)
        return m + torch.eye(self.k, device=x.device, dtype=m.dtype).unsqueeze(0)


class PointNetSeg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: float = 1.0,
                 feature_tnet: bool = False, dropout: float = 0.3, norm: str = "bn"):
        super().__init__()
        NORM["type"] = norm
        w = lambda c: max(8, int(c * width))  # noqa: E731
        self.local = _mlp([in_channels, w(64), w(64)])
        self.tnet = TNet(w(64), width) if feature_tnet else None
        self.globl = _mlp([w(64), w(64), w(128), w(1024)])
        self.head = nn.Sequential(
            _mlp([w(64) + w(1024), w(512), w(256), w(128)], dropout=dropout),
            nn.Conv1d(w(128), num_classes, 1))

    def forward(self, x):
        n = x.shape[2]
        local = self.local(x)
        trans = None
        if self.tnet is not None:
            trans = self.tnet(local)
            local = torch.bmm(trans.transpose(1, 2), local)
        g = self.globl(local).max(dim=2, keepdim=True).values
        out = self.head(torch.cat([local, g.expand(-1, -1, n)], 1))
        return out, trans


@torch.no_grad()
def recalibrate_bn(model, batches, device):
    """Re-estimate BatchNorm running mean/var as an exact average over `batches`."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative average
    was_training = model.training
    model.train()
    for x in batches:
        model(x.to(device, non_blocking=True))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.train(was_training)


def tnet_regularizer(trans):
    """||I - A A^T||^2, the PointNet feature-transform regulariser."""
    if trans is None:
        return 0.0
    k = trans.shape[1]
    eye = torch.eye(k, device=trans.device, dtype=trans.dtype).unsqueeze(0)
    return ((torch.bmm(trans, trans.transpose(1, 2)) - eye) ** 2).sum(dim=(1, 2)).mean()
