import typing as tp

import torch
import torch.nn as nn

from ...config import ModuleConfig
from ...utils.misc import create_instance


class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()

    def forward(self, pred, label):
        vx = pred - torch.mean(pred)
        vy = label - torch.mean(label)

        cost = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
        )

        return -cost


class PairwiseRankingLoss(nn.Module):
    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_a = pred.unsqueeze(1)
        pred_b = pred.unsqueeze(0)
        pred_diff = pred_a - pred_b
        target_a = target.unsqueeze(1)
        target_b = target.unsqueeze(0)
        target_diff = target_a - target_b

        ranking_prod = -(pred_diff * target_diff)
        ranking_prod = torch.clamp(ranking_prod, min=0)
        eye_mask = (
            ~torch.eye(
                ranking_prod.shape[0],
                device=ranking_prod.device,
                dtype=torch.bool,
            )
        ).int()
        ranking_prod = ranking_prod * eye_mask

        loss = ranking_prod.sum()
        return loss


class CompositeLoss(nn.Module):
    def __init__(
        self, components: tp.List[ModuleConfig], weights: tp.List[float]
    ) -> None:
        super().__init__()
        self.components = []
        for component in components:
            self.components.append(create_instance(component))
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for component in self.components:
            losses.append(component(pred, target))
        loss = sum(
            [loss * weight for loss, weight in zip(losses, self.weights)]
        )
        return loss
