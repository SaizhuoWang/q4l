# Code referred from Qlib
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class FinetuneModel(nn.Module):
    """FinuetuneModel for adding a layer by the end."""

    def __init__(self, input_dim, output_dim, trained_model):
        super().__init__()
        self.model = trained_model
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, priors):
        return self.fc(self.model(x, priors)[0]).squeeze()  # take the vec out


class DecoderStep(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, out_dim, shared, n_ind, vbs)
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.fea_tran(x)
        return self.fc(x)


class TabNet_Decoder(nn.Module):
    def __init__(self, inp_dim, out_dim, n_shared, n_ind, vbs, n_steps):
        """TabNet decoder that is used in pre-training."""
        super().__init__()
        self.out_dim = out_dim
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * out_dim))
            for x in range(n_shared - 1):
                self.shared.append(
                    nn.Linear(out_dim, 2 * out_dim)
                )  # preset the linear function we will use
        else:
            self.shared = None
        self.n_steps = n_steps
        self.steps = nn.ModuleList()
        for x in range(n_steps):
            self.steps.append(
                DecoderStep(inp_dim, out_dim, self.shared, n_ind, vbs)
            )

    def forward(self, x):
        out = torch.zeros(x.size(0), self.out_dim).to(x.device)
        for step in self.steps:
            out += step(x)
        return out


class TabNet(nn.Module):
    def __init__(
        self,
        inp_dim=6,
        out_dim=6,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        relax=1.2,
        vbs=1024,
    ):
        """TabNet AKA the original encoder.

        Args:
            n_d: dimension of the features used to calculate the final results
            n_a: dimension of the features input to the attention transformer of the next step
            n_shared: numbr of shared steps in feature transformer(optional)
            n_ind: number of independent steps in feature transformer
            n_steps: number of steps of pass through tabbet
            relax coefficient:
            virtual batch size:

        """
        super().__init__()

        # set the number of shared step in feature transformer
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(
                    nn.Linear(n_d + n_a, 2 * (n_d + n_a))
                )  # preset the linear function we will use
        else:
            self.shared = None

        self.first_step = FeatureTransformer(
            inp_dim, n_d + n_a, self.shared, n_ind, vbs
        )
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            self.steps.append(
                DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs)
            )
        self.fc = nn.Linear(n_d, out_dim)
        self.bn = nn.BatchNorm1d(inp_dim, momentum=0.01)
        self.n_d = n_d

    def forward(self, x, priors):
        assert not torch.isnan(x).any()
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d :]
        sparse_loss = []
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        for step in self.steps:
            x_te, loss = step(x, x_a, priors)
            out += F.relu(
                x_te[:, : self.n_d]
            )  # split the feature from feat_transformer
            x_a = x_te[:, self.n_d :]
            sparse_loss.append(loss)
        return self.fc(out), sum(sparse_loss)


class GBN(nn.Module):
    """Ghost Batch Normalization an efficient way of doing batch normalization.

    Args:
        vbs: virtual batch size

    """

    def __init__(self, inp, vbs=1024, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        if x.size(0) <= self.vbs:  # can not be chunked
            return self.bn(x)
        else:
            chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
            res = [self.bn(y) for y in chunk]
            return torch.cat(res, 0)


class GLU(nn.Module):
    """GLU block that extracts only the most essential information.

    Args:
        vbs: virtual batch size

    """

    def __init__(self, inp_dim, out_dim, fc=None, vbs=1024):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        return torch.mul(x[:, : self.od], torch.sigmoid(x[:, self.od :]))


class AttentionTransformer(nn.Module):
    """
    Args:
        relax: relax coefficient. The greater it is, we can
        use the same features more. When it is set to 1
        we can use every feature only once
    """

    def __init__(self, d_a, inp_dim, relax, vbs=1024):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax

    # a:feature from previous decision step
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = SparsemaxFunction.apply(a * priors)
        priors = priors * (self.r - mask)  # updating the prior
        return mask


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = float(np.sqrt(0.5))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class DecisionStep(nn.Module):
    """One step for the TabNet."""

    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs):
        super().__init__()
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
        self.fea_tran = FeatureTransformer(
            inp_dim, n_d + n_a, shared, n_ind, vbs
        )

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, sparse_loss


def make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """SparseMax function for replacing reLU."""

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction.threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def threshold_and_support(input, dim=-1):
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


class TabNetModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        relax=1.2,
        vbs=1024,
    ):
        super().__init__()
        self.tabnet = TabNet(
            input_dim,
            output_dim,
            n_d=n_d,
            n_a=n_a,
            n_shared=n_shared,
            n_ind=n_ind,
            n_steps=n_steps,
            relax=relax,
            vbs=vbs,
        )
        self.decoder = TabNet_Decoder(
            output_dim, input_dim, n_shared, n_ind, vbs, n_steps
        )

    def forward(self, x: torch.Tensor):
        # Prior is bernoulli distribution, infer the shape from x
        priors = 1 - torch.bernoulli(
            torch.ones(x.shape[0], x.shape[1]) * 0.5
        ).to(x.device)
        x, sparse_loss = self.tabnet(x, priors)
        x = self.decoder(x)
        return x, sparse_loss
