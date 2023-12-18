import typing as tp

import torch.nn as nn


class BaseHeadModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: tp.Dict, batch_idx: int):
        pass
