import typing as tp

import torch
import torch.nn as nn

from ...base import EMBEDDING_KEY, INPUT_KEY


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(
        self, batch: tp.Dict[str, torch.Tensor]
    ) -> tp.Dict[str, torch.Tensor]:

        x = batch[INPUT_KEY].squeeze()
        return {EMBEDDING_KEY: x}
