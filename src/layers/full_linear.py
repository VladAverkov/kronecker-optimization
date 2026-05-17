"""Standard nn.Linear wrapper for baseline comparison."""

import torch.nn as nn


class FullLinear(nn.Module):
    """Plain nn.Linear. Accepts (and ignores) n_terms for API compatibility."""

    def __init__(self, in_features, out_features, n_terms=1, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
