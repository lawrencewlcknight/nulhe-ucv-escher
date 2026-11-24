"""Diagnostic-only sequential street specialists; no production loader changes."""
import torch
from torch import nn

from .features import StructuredHULHEMLP
from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear


class ResidualTransform(nn.Module):
    def __init__(self, width, inner):
        super().__init__()
        self.path = nn.Sequential(
            nn.LayerNorm(width, eps=1e-5), nn.ReLU(), SonnetLinear(width, inner, False),
            nn.LayerNorm(inner, eps=1e-5), nn.ReLU(), SonnetLinear(inner, width, False))

    def forward(self, x):
        return x + .5 * self.path(x)


class SequentialStreetNetwork(StructuredHULHEMLP):
    """All four specialists execute, even when their parameters are frozen.

    Freeze parameters, never activations: early specialists must receive gradients
    through the later stages. No street gating or hidden opponent inputs.
    """
    def __init__(self, layout):
        super().__init__(layout, [384], 3, branch_width=128)
        self.shared_stages = nn.ModuleList([ResidualTransform(384, 192) for _ in range(4)])
        self.specialists = nn.ModuleList([ResidualTransform(384, 224) for _ in range(4)])
        self.output = ZeroInitLinear(768, 3, False)
        self.checkpoint_model_type = 'diagnostic_sequential_streets_v1'

    def forward(self, x):
        split = self.layout.card_size + self.layout.strategic_size
        joined = torch.cat([self.card_encoder(x[..., :self.layout.card_size]),
                            self.context_encoder(x[..., split:]),
                            self.strategic_encoder(x[..., self.layout.card_size:split])], -1)
        entrance = self.trunk(joined)
        hidden = entrance
        for shared, specialist in zip(self.shared_stages, self.specialists):
            hidden = specialist(shared(hidden))
        return self.output(torch.cat([entrance, hidden], -1))

    def trainable_street(self, street=None):
        """None trains everything. Shared parameters always remain trainable."""
        if street is not None and street not in range(4):
            raise ValueError('Invalid street')
        for p in self.parameters():
            p.requires_grad_(True)
        for index, specialist in enumerate(self.specialists):
            for p in specialist.parameters():
                p.requires_grad_(street is None or index == street)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (SonnetLinear, ZeroInitLinear, nn.LayerNorm)):
                module.reset_parameters()

    def checkpoint_metadata(self):
        return dict(type=self.checkpoint_model_type, layout=self.layout.to_dict(),
                    branch_width=128, width=384, shared_inner=192, specialist_inner=224,
                    stages=4, residual_scale=.5, parameters=sum(p.numel() for p in self.parameters()),
                    diagnostic_only=True)
