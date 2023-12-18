import typing as tp

import torch.nn as nn

from ....utils.misc import create_instance
from .base import BaseTemporalEncoder


class MLPTSEncoder(BaseTemporalEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: tp.List[int],
        output_dim: int,
        window_size: int,
        activation: tp.Dict,
    ) -> None:
        """An MLP-based time-series encoder. Suppose the input is of shape
        (batch_size, seq_len, input_size), input is first.

        flattened to (batch_size, seq_len * input_size), then fed into a MLP.
        MLP may have multiple layers, and the output of the last layer is used as the
        output embedding.

        Parameters
        ----------
        input_size : int
            The dimension of the input time-series.
        hidden_size : tp.List[int]
            A list of hidden layer sizes.
        window_size : int
            The window size.
        activation : str
            The activation function.

        """
        super().__init__()
        self.window_size = window_size
        self.activation = create_instance(activation)
        self.layers = nn.Sequential()
        num_layers = 1 + len(hidden_size)
        input_dim = input_size * window_size
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            in_dim = input_dim
            for hidden_dim in hidden_size:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
                self.layers.append(self.activation)
                in_dim = hidden_dim
            self.layers.append(nn.Linear(in_dim, output_dim))

    def forward(self, input_data: tp.Dict) -> tp.Dict:
        x = input_data["x"].squeeze(0)
        x = x.reshape(x.shape[0], -1)
        emb = self.layers(x)
        return {"emb": emb}
