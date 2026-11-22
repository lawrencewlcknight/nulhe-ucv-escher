"""Versioned, opt-in residual architecture for the fresh deep32 experiment."""
from torch import nn

from .features import StructuredHULHEMLP
from vr_deep_cfr.solver import SonnetLinear


class PreActivationResidualBlock(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.path = nn.Sequential(
            nn.LayerNorm(width, eps=1e-5), nn.ReLU(), SonnetLinear(width, width, False),
            nn.LayerNorm(width, eps=1e-5), nn.ReLU(), SonnetLinear(width, width, False))

    def forward(self, inputs):
        return inputs + 0.5 * self.path(inputs)


class DeepResidualHULHEMLP(StructuredHULHEMLP):
    """128-wide branches, 256 stem, four two-layer blocks, 256 readout.

    Ten hidden linear layers exclude the input branches and zero output head.
    Version fixes normalization, activation placement and residual scale as well
    as tensor shapes, so playable policies cannot silently change semantics.
    """
    def __init__(self, layout, hidden_layers, output_size, *, branch_width):
        if list(hidden_layers) != [256] * 10 or branch_width != 128:
            raise ValueError('deep_residual_v1 requires ten 256-wide layers and 128-wide branches')
        super().__init__(layout, [256], output_size, branch_width=branch_width)
        self.hidden_layers = [256] * 10
        self.checkpoint_model_type = 'structured_hulhe_deep_residual_v1'
        stem = self.trunk[0]
        self.trunk = nn.Sequential(stem, *[PreActivationResidualBlock() for _ in range(4)],
            nn.LayerNorm(256, eps=1e-5), nn.ReLU(), SonnetLinear(256, 256, True))

    def reset_parameters(self):
        for branch in ('card_encoder', 'context_encoder', 'strategic_encoder'):
            if hasattr(self, branch):
                getattr(self, branch).reset_parameters()
        for module in self.trunk.modules():
            if isinstance(module, (SonnetLinear, nn.LayerNorm)):
                module.reset_parameters()
        self.output.reset_parameters()

    def checkpoint_metadata(self):
        return dict(super().checkpoint_metadata(), residual_blocks=4,
                    residual_scale=0.5, layer_norm_eps=1e-5, preactivation=True)
