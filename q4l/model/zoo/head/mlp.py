import typing as tp
from typing import Dict, List

import torch.nn as nn

from ...base import EMBEDDING_KEY, PREDICTION_KEY


# MLP Layers
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expansion_coefficients: List[float],
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        num_layers = 1 + len(expansion_coefficients)
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            in_dim = input_dim
            for coeff in expansion_coefficients:
                hidden_dim = int(coeff * in_dim)
                self.layers.append(nn.Linear(in_dim, hidden_dim))

                # NOTE: Currently we use ReLU as the default activation
                # function, add more options later
                self.layers.append(nn.ReLU())

                in_dim = hidden_dim
            self.layers.append(nn.Linear(in_dim, output_dim))

    def get_softmax(self, batch: Dict) -> tp.Dict:
        emb = batch[EMBEDDING_KEY]
        for layer in self.layers[:-1]:
            emb = layer(emb)
        return {"logits": emb}

    def forward(self, batch: Dict) -> tp.Dict:
        emb = batch[EMBEDDING_KEY]
        pred = self.layers(emb)
        return {PREDICTION_KEY: pred}
