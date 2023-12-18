import typing as tp
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...base import EMBEDDING_KEY, PREDICTION_KEY


class Gate(nn.Module):
    def __init__( self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden_dim = input_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, (1, 3))
        self.conv2 = nn.Conv2d(input_dim, hidden_dim, (1, 3))
        self.conv3 = nn.Conv2d(input_dim, hidden_dim, (1, 3))
        self.fully = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch: Dict) -> tp.Dict:
        emb = batch[EMBEDDING_KEY]
        emb = emb.permute(0, 3, 1, 2)
        temp = self.conv1(emb) + torch.sigmoid(self.conv2(emb))
        out = F.relu(temp + self.conv3(emb))

        out = out.permute(0, 2, 3, 1)
        pred = self.fully(out.reshape((out.shape[0], out.shape[1], -1)))
        return {PREDICTION_KEY: pred}
