import typing as tp

import torch
import torch.nn as nn
import torch.nn.init as init

from q4l.model.zoo.temporal.base import BaseTemporalEncoder


class SFM_Model(nn.Module):
    def __init__(
        self,
        d_feat=6,
        output_dim=1,
        freq_dim=10,
        hidden_size=64,
        dropout_W=0.0,
        dropout_U=0.0,
    ):
        super().__init__()

        self.input_dim = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size

        self.W_i = nn.Parameter(
            init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim)))
        )
        self.U_i = nn.Parameter(
            init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim))
        )
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(
            init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim))
        )
        self.U_ste = nn.Parameter(
            init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim))
        )
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(
            init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim))
        )
        self.U_fre = nn.Parameter(
            init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim))
        )
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(
            init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim))
        )
        self.U_c = nn.Parameter(
            init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim))
        )
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(
            init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim))
        )
        self.U_o = nn.Parameter(
            init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim))
        )
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(
            init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim))
        )
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc_out = nn.Linear(self.output_dim, 1)

        self.states = []

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        time_step = input.shape[1]
        self.device = input.device

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:  # hasn't initialized yet
                self.init_states(x)
            p_tm1 = self.states[0]  # noqa: F841
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            frequency = (
                torch.arange(0, self.freq_dim, device=self.device)
                / self.freq_dim
            )

            x_i = torch.matmul(x, self.W_i) + self.b_i
            x_ste = torch.matmul(x, self.W_ste) + self.b_ste
            x_fre = torch.matmul(x, self.W_fre) + self.b_fre
            x_c = torch.matmul(x, self.W_c) + self.b_c
            x_o = torch.matmul(x, self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1, self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1, self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1, self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1, self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * torch.pi) * time * frequency

            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A, self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)

            o = self.inner_activation(x_o + torch.matmul(h_tm1, self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []
        return p

    def init_states(self, x):
        reducer_f = torch.zeros(
            (self.hidden_dim, self.freq_dim), device=self.device
        )
        reducer_p = torch.zeros(
            (self.hidden_dim, self.output_dim), device=self.device
        )

        init_state_h = torch.zeros(self.hidden_dim, device=self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h, device=self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0, device=self.device)

        self.states = [
            init_state_p,
            init_state_h,
            init_state_S_re,
            init_state_S_im,
            init_state_time,
            None,
            None,
            None,
        ]


class SFM(BaseTemporalEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_dim: int,
        freq_dim: int = 10,
        dropout_W: float = 0.0,
        dropout_U: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = SFM_Model(
            d_feat=input_size,
            hidden_size=hidden_size,
            output_dim=output_dim,
            freq_dim=freq_dim,
            dropout_W=dropout_W,
            dropout_U=dropout_U,
        )

    def forward(self, input_data: tp.Dict) -> tp.Dict:
        x = input_data["x"].squeeze(0)
        emb = self.model(x)
        return {"emb": emb}
