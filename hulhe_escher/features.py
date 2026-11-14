"""Suit-canonical cards with street identity and native FCPA betting history.

The policy sees only its own cards and public events. Critics additionally see
both players' hole cards. Flop, turn and river occupy separate channels; merging
all five public cards would discard information about earlier decisions.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .game import load_hulhe_game
from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear

ENCODER_ID = "hulhe_suit_canonical_fcpa_v1"


@dataclass(frozen=True)
class FeatureLayout:
    name: str
    total_size: int
    card_size: int

    @property
    def context_size(self):
        return self.total_size - self.card_size

    def to_dict(self):
        return dict(name=self.name, total_size=self.total_size, card_size=self.card_size)


def _canonicalise_suits(*channels):
    signatures = [tuple(float(x) for channel in channels for x in channel[suit])
                  for suit in range(4)]
    order = sorted(range(4), key=lambda suit: signatures[suit], reverse=True)
    return tuple(channel[order].reshape(-1) for channel in channels)


class HULHEFeatureEncoder:
    encoder_id = ENCODER_ID

    def __init__(self):
        game = load_hulhe_game()
        self.raw_size = game.information_state_tensor_size()
        self.history_slots = game.max_game_length()
        if self.raw_size != 106 + 3 * self.history_slots:
            raise ValueError("OpenSpiel FCPA tensor schema changed")
        # Two player bits, native betting-history suffix, street, legal mask.
        context_size = 2 + self.raw_size - 106 + 4 + 4
        self.policy_layout = FeatureLayout("player_information_state", 208 + context_size, 208)
        self.full_state_layout = FeatureLayout("critic_full_state", 260 + context_size, 260)

    @property
    def policy_size(self):
        return self.policy_layout.total_size

    @property
    def full_state_size(self):
        return self.full_state_layout.total_size

    def metadata(self):
        return {"id": self.encoder_id, "version": 1,
                "raw_open_spiel_size": self.raw_size,
                "policy_layout": self.policy_layout.to_dict(),
                "full_state_layout": self.full_state_layout.to_dict(),
                "board_channels": ["flop", "turn", "river"],
                "bet_amount_scale_chips": 20000}

    def _raw(self, state, player):
        raw = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        if raw.shape != (self.raw_size,):
            raise ValueError("State does not match the versioned HULHE tensor layout")
        return raw

    def _board(self, state):
        # Chance events are distinguished by their player id, never by action
        # number (card ids and betting action ids overlap).
        cards = [event.action for event in state.full_history() if event.player == -1][4:]
        if len(cards) > 5:
            raise ValueError("Too many public cards")
        channels = [np.zeros((4, 13), dtype=np.float32) for _ in range(3)]
        for index, card in enumerate(cards):
            street = 0 if index < 3 else index - 2
            channels[street][card % 4, card // 4] = 1
        return channels, len(cards)

    def _context(self, state, raw, board_count):
        street = np.zeros(4, dtype=np.float32)
        street[0 if board_count == 0 else min(3, board_count - 2)] = 1
        mask = np.asarray(state.legal_actions_mask(), dtype=np.float32)
        if mask.shape != (4,):
            raise ValueError("Encoder accepts the four-action FCPA game only")
        betting = raw[106:].copy()
        # Native history stores two action bits per slot followed by one chip
        # amount per slot. Keep the bits unchanged and scale monetary inputs.
        betting[2 * self.history_slots:] /= 20000.0
        return np.concatenate((raw[:2], betting, street, mask))

    def information_state(self, state, player=None):
        player = state.current_player() if player is None else int(player)
        if player not in (0, 1) or state.is_chance_node() or state.is_terminal():
            raise ValueError("Encode decision states only")
        raw = self._raw(state, player)
        board, count = self._board(state)
        hole = raw[2:54].reshape(13, 4).T
        return np.concatenate((*_canonicalise_suits(hole, *board),
                               self._context(state, raw, count))).astype(np.float32)

    def full_state(self, state):
        if state.is_terminal():
            # Terminal replay rows carry the payoff and done mask; TD never
            # bootstraps from this placeholder.
            return np.zeros(self.full_state_size, dtype=np.float32)
        player = state.current_player()
        if player not in (0, 1):
            raise ValueError("Encode decision states only")
        raw = [self._raw(state, p) for p in (0, 1)]
        board, count = self._board(state)
        holes = [r[2:54].reshape(13, 4).T for r in raw]
        return np.concatenate((*_canonicalise_suits(*holes, *board),
                               self._context(state, raw[player], count))).astype(np.float32)


def make_feature_encoder(encoder_id=ENCODER_ID):
    if encoder_id != ENCODER_ID:
        raise ValueError(f"Unsupported HULHE encoder: {encoder_id}")
    return HULHEFeatureEncoder()


def encoder_from_metadata(metadata):
    if metadata is None:
        raise ValueError("A HULHE policy must declare its encoder")
    encoder = make_feature_encoder(metadata.get("id"))
    if metadata != encoder.metadata():
        raise ValueError("Checkpoint encoder differs from the versioned HULHE contract")
    return encoder


class StructuredHULHEMLP(nn.Module):
    """Small CPU-friendly two-branch MLP for cards and betting context."""

    checkpoint_model_type = "structured_hulhe_mlp_v1"

    def __init__(
        self,
        layout: FeatureLayout,
        hidden_layers,
        output_size: int,
        *,
        branch_width: int,
    ):
        super().__init__()
        hidden_layers = [int(width) for width in hidden_layers]
        if not hidden_layers or int(branch_width) <= 0:
            raise ValueError("Structured HULHE MLP requires positive hidden widths")
        self.layout = layout
        self.input_size = int(layout.total_size)
        self.output_size = int(output_size)
        self.hidden_layers = hidden_layers
        self.branch_width = int(branch_width)
        self.card_encoder = SonnetLinear(
            layout.card_size, self.branch_width, activation=True
        )
        self.context_encoder = SonnetLinear(
            layout.context_size, self.branch_width, activation=True
        )
        widths = [2 * self.branch_width, *hidden_layers]
        self.trunk = nn.Sequential(
            *[
                SonnetLinear(input_width, output_width, activation=True)
                for input_width, output_width in zip(widths[:-1], widths[1:])
            ]
        )
        self.output = ZeroInitLinear(hidden_layers[-1], self.output_size, False)

    def reset_parameters(self) -> None:
        self.card_encoder.reset_parameters()
        self.context_encoder.reset_parameters()
        for layer in self.trunk:
            layer.reset_parameters()
        self.output.reset_parameters()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "type": self.checkpoint_model_type,
            "layout": self.layout.to_dict(),
            "hidden_layers": list(self.hidden_layers),
            "branch_width": self.branch_width,
            "output_size": self.output_size,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cards = inputs[..., : self.layout.card_size]
        context = inputs[..., self.layout.card_size :]
        joined = torch.cat(
            [self.card_encoder(cards), self.context_encoder(context)], dim=-1
        )
        return self.output(self.trunk(joined))
