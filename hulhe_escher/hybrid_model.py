"""Versioned wide-to-deep residual architecture for fresh hybrid32 training."""
import torch
from torch import nn

from .features import StructuredHULHEMLP
from .deep_model import PreActivationResidualBlock
from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear

# Longest hidden linear path, excluding input branches and the action head.
HYBRID_LAYERS = [512, 256, 512] + [256] * 10


class WideResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = nn.Sequential(
            nn.LayerNorm(512, eps=1e-5), nn.ReLU(), SonnetLinear(512, 256, False),
            nn.LayerNorm(256, eps=1e-5), nn.ReLU(), SonnetLinear(256, 512, False))

    def forward(self, inputs):
        return inputs + 0.5 * self.path(inputs)


class HybridResidualHULHEMLP(StructuredHULHEMLP):
    def __init__(self, layout, hidden_layers, output_size, *, branch_width):
        if list(hidden_layers) != HYBRID_LAYERS or branch_width != 128 or not layout.strategic_size:
            raise ValueError('hybrid_residual_v1 requires the versioned hybrid trunk and three 128-wide branches')
        super().__init__(layout, [512], output_size, branch_width=branch_width)
        self.hidden_layers = list(HYBRID_LAYERS)
        self.checkpoint_model_type = 'structured_hulhe_hybrid_residual_v1'
        self.wide = nn.Sequential(self.trunk[0], WideResidualBlock())
        self.trunk = nn.Sequential(SonnetLinear(512, 256, True),
            *[PreActivationResidualBlock() for _ in range(4)],
            nn.LayerNorm(256, eps=1e-5), nn.ReLU(), SonnetLinear(256, 256, True))
        self.output = ZeroInitLinear(768, output_size, False)

    def reset_parameters(self):
        # Reset each leaf exactly once, including affine normalization parameters.
        for module in self.modules():
            if isinstance(module, (SonnetLinear, ZeroInitLinear, nn.LayerNorm)):
                module.reset_parameters()

    def forward(self, inputs):
        split = self.layout.card_size + self.layout.strategic_size
        joined = torch.cat([
            self.card_encoder(inputs[..., :self.layout.card_size]),
            self.context_encoder(inputs[..., split:]),
            self.strategic_encoder(inputs[..., self.layout.card_size:split])], dim=-1)
        wide = self.wide(joined)
        deep = self.trunk(wide)
        return self.output(torch.cat([wide, deep], dim=-1))

    def checkpoint_metadata(self):
        return dict(super().checkpoint_metadata(), wide_residual_layers=[512, 256, 512],
            deep_residual_blocks=4, residual_scale=0.5, layer_norm_eps=1e-5,
            preactivation=True, output_features=['wide_512', 'deep_256'])
