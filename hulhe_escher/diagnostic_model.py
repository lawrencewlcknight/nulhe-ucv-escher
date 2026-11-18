"""Opt-in policy capacity candidate; production trainers keep their architecture."""
import torch
from torch import nn
from .features import StructuredHULHEMLP


class ResidualPolicy(StructuredHULHEMLP):
    def __init__(self, layout, *, width=384, blocks=2, head=256, branch_width=128):
        super().__init__(layout, [width], 3, branch_width=branch_width)
        self.width, self.blocks, self.head = width, blocks, head
        self.residual = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                      nn.Linear(width, width)) for _ in range(blocks)])
        self.head_layer = nn.Linear(width, head)
        self.output = nn.Linear(head, 3)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'residual'):
            for block in self.residual:
                for layer in block:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        nn.init.zeros_(layer.bias)
                # Identity start preserves gradients through the deeper candidate.
                nn.init.zeros_(block[-1].weight)
            nn.init.kaiming_normal_(self.head_layer.weight, nonlinearity='relu')
            nn.init.zeros_(self.head_layer.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        split = self.layout.card_size + self.layout.strategic_size
        branches = [self.card_encoder(inputs[..., :self.layout.card_size]),
                    self.context_encoder(inputs[..., split:])]
        if self.layout.strategic_size:
            branches.append(self.strategic_encoder(inputs[..., self.layout.card_size:split]))
        x = self.trunk(torch.cat(branches, -1))
        for block in self.residual:
            x = torch.relu(x + block(x))
        return self.output(torch.relu(self.head_layer(x)))

    def checkpoint_metadata(self):
        return dict(type='hulhe_diagnostic_residual_v1', layout=self.layout.to_dict(),
                    width=self.width, blocks=self.blocks, head=self.head,
                    branch_width=self.branch_width, output_size=3)
