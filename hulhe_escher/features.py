"""Suit-canonical cards with street identity and fixed-limit betting history.

The policy sees only its own cards and public events. Critics additionally see
both players' hole cards. Flop, turn and river occupy separate channels; merging
all five public cards would discard information about earlier decisions.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .game import GAME_CONTRACT_ID, NUM_ACTIONS, load_hulhe_game, validate_hulhe_game
from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear

LEGACY_ENCODER_ID = "hulhe_suit_canonical_limit_v2"
ENCODER_ID = "hulhe_strategic_limit_v3"


@dataclass(frozen=True)
class FeatureLayout:
    name: str
    total_size: int
    card_size: int
    strategic_size: int = 0

    @property
    def context_size(self):
        return self.total_size - self.card_size - self.strategic_size

    def to_dict(self):
        result = dict(name=self.name, total_size=self.total_size, card_size=self.card_size)
        if self.strategic_size:
            result["strategic_size"] = self.strategic_size
        return result


def _canonicalise_suits(*channels):
    signatures = [tuple(float(x) for channel in channels for x in channel[suit])
                  for suit in range(4)]
    order = sorted(range(4), key=lambda suit: signatures[suit], reverse=True)
    return tuple(channel[order].reshape(-1) for channel in channels)


class LegacyHULHEFeatureEncoder:
    encoder_id = LEGACY_ENCODER_ID

    def __init__(self):
        game = load_hulhe_game()
        self._validated_game = game
        self.raw_size = game.information_state_tensor_size()
        self.history_slots = game.max_game_length()
        if (self.history_slots != 34 or self.raw_size != 208
                or game.num_distinct_actions() != NUM_ACTIONS):
            raise ValueError("OpenSpiel fixed-limit tensor schema changed")
        # Two player bits, two action bits per history slot, street, legal mask.
        # Native sizing channels are identically zero in the limit game.
        context_size = 2 + 2 * self.history_slots + 4 + NUM_ACTIONS
        self.policy_layout = FeatureLayout("player_information_state", 208 + context_size, 208)
        self.full_state_layout = FeatureLayout("critic_full_state", 260 + context_size, 260)

    @property
    def policy_size(self):
        return self.policy_layout.total_size

    @property
    def full_state_size(self):
        return self.full_state_layout.total_size

    def metadata(self):
        return {"id": self.encoder_id, "version": 2,
                "game_contract_id": GAME_CONTRACT_ID,
                "raw_open_spiel_size": self.raw_size,
                "history_slots": self.history_slots,
                "policy_layout": self.policy_layout.to_dict(),
                "full_state_layout": self.full_state_layout.to_dict(),
                "board_channels": ["flop", "turn", "river"],
                "betting_history": "two native action bits per slot; zero sizing channels omitted",
                "num_actions": NUM_ACTIONS}

    def _validate_state_game(self, state):
        game = state.get_game()
        if game is not self._validated_game:
            validate_hulhe_game(game)
            self._validated_game = game

    def _raw(self, state, player):
        self._validate_state_game(state)
        raw = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        if raw.shape != (self.raw_size,):
            raise ValueError("State does not match the versioned HULHE tensor layout")
        if np.any(raw[106 + 2 * self.history_slots:]):
            raise ValueError("Unexpected nonzero fixed-limit sizing channels")
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
        if mask.shape != (NUM_ACTIONS,):
            raise ValueError("Encoder accepts the three-action fixed-limit game only")
        betting = raw[106:106 + 2 * self.history_slots]
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
        self._validate_state_game(state)
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


class HULHEFeatureEncoder(LegacyHULHEFeatureEncoder):
    """Exact information state plus strategic summaries; no lossy bucketing."""
    encoder_id = ENCODER_ID

    def __init__(self):
        super().__init__()
        from .card_features import CARD_NAMES
        from .betting_features import BETTING_NAMES
        self.policy_layout = FeatureLayout("strategic_player_v3",
            285 + len(CARD_NAMES) + len(BETTING_NAMES), 208, len(CARD_NAMES))
        self.full_state_layout = FeatureLayout("strategic_critic_v3",
            337 + 2*len(CARD_NAMES) + 2*len(BETTING_NAMES) + 4,
            260, 2*len(CARD_NAMES) + 4)

    def metadata(self):
        from .card_features import feature_spec
        from .betting_features import BETTING_NAMES
        return {**super().metadata(), "version": 3,
                "base_encoder_id": LEGACY_ENCODER_ID,
                "strategic_cards": feature_spec(), "betting_features": list(BETTING_NAMES),
                "feature_order": "canonical cards; strategic cards; player/history/street; economics; legal mask",
                "critic_extra_order": "player0, player1, exact current comparison; economics player0 then player1",
                "critic_comparison": ["available", "player0_current_win", "current_tie", "player1_current_win"],
                "public_cache_entries": 4096, "critic_comparison_cache_entries": 8192,
                "normalisation": "bounded [0,1]; pot/4800, commitments/2400, round/800, cost/200; signed interactions shifted",
                "information_preservation": "all v2 exact channels retained; no buckets or action overrides"}

    @staticmethod
    def _ids(channel):
        suits, ranks = np.nonzero(channel)
        return tuple(sorted(map(int, 4*ranks+suits)))

    def information_state(self, state, player=None):
        from .card_features import card_features
        from .betting_features import public_rounds, betting_features
        player = state.current_player() if player is None else int(player)
        if player not in (0,1) or state.is_chance_node() or state.is_terminal():
            raise ValueError("Encode decision states only")
        raw = self._raw(state, player)
        board, count = self._board(state)
        hole = raw[2:54].reshape(13,4).T
        strategic = card_features(self._ids(hole), self._ids(sum(board)))
        context = self._context(state, raw, count)
        economics = betting_features(public_rounds(state), player, strategic, actor=state.current_player())
        return np.concatenate((*_canonicalise_suits(hole,*board), strategic,
                               context[:-3], economics, context[-3:])).astype(np.float32)

    def full_state(self, state):
        from .card_features import card_features, critic_comparison
        from .betting_features import public_rounds, betting_features
        self._validate_state_game(state)
        if state.is_terminal():
            return np.zeros(self.full_state_size, dtype=np.float32)
        player = state.current_player()
        if player not in (0,1): raise ValueError("Encode decision states only")
        raw = [self._raw(state,p) for p in (0,1)]
        holes = [r[2:54].reshape(13,4).T for r in raw]
        board, count = self._board(state)
        public = self._ids(sum(board))
        private = [self._ids(h) for h in holes]
        # Each perspective must canonicalise independently of the other hand.
        strategic = [card_features(h,public) for h in private]
        comparison = critic_comparison(*private, public)
        rounds = public_rounds(state)
        economics = [betting_features(rounds,p,strategic[p],actor=player) for p in (0,1)]
        context = self._context(state,raw[player],count)
        return np.concatenate((*_canonicalise_suits(*holes,*board), *strategic, comparison,
                               context[:-3], *economics, context[-3:])).astype(np.float32)


def make_feature_encoder(encoder_id=ENCODER_ID):
    if encoder_id == LEGACY_ENCODER_ID:
        return LegacyHULHEFeatureEncoder()
    if encoder_id == ENCODER_ID:
        return HULHEFeatureEncoder()
    raise ValueError(f"Unsupported HULHE encoder: {encoder_id}")


def encoder_from_metadata(metadata):
    if metadata is None:
        raise ValueError("A HULHE policy must declare its encoder")
    encoder = make_feature_encoder(metadata.get("id"))
    if metadata != encoder.metadata():
        raise ValueError("Checkpoint encoder differs from the versioned HULHE contract")
    return encoder


class StructuredHULHEMLP(nn.Module):
    """Separate exact-card, strategic-card and betting-context branches.

    Legacy layouts retain their original two-branch model for evaluation.
    """

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
        self.checkpoint_model_type = ("structured_hulhe_mlp_v2" if layout.strategic_size
                                      else "structured_hulhe_mlp_v1")
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
        if layout.strategic_size:
            self.strategic_encoder = SonnetLinear(layout.strategic_size, self.branch_width, activation=True)
        widths = [(3 if layout.strategic_size else 2) * self.branch_width, *hidden_layers]
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
        if self.layout.strategic_size:
            self.strategic_encoder.reset_parameters()
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
        split = self.layout.card_size + self.layout.strategic_size
        context = inputs[..., split:]
        branches = [self.card_encoder(cards), self.context_encoder(context)]
        if self.layout.strategic_size:
            branches.append(self.strategic_encoder(inputs[..., self.layout.card_size:split]))
        joined = torch.cat(branches, dim=-1)
        return self.output(self.trunk(joined))
