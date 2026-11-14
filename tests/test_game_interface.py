"""Rules and learner-boundary regression tests for the Fell Omen 2 game."""
from copy import deepcopy
import pickle

import numpy as np
import pyspiel
import pytest

from hulhe_escher.checkpointing import (
    CHECKPOINT_TYPE, CHECKPOINT_VERSION, LoadedHULHEPolicy, load_checkpoint_payload,
)
from hulhe_escher.features import HULHEFeatureEncoder, make_feature_encoder
from hulhe_escher.game import (
    HULHE_GAME_PARAMETERS, load_hulhe_game, serialisable_game_definition,
)
from hulhe_escher.training_state import restore_training_state


def decision_state(street=0, game=None):
    state = (load_hulhe_game() if game is None else game).new_initial_state()
    for card in (48, 49, 0, 1):  # AA versus 22, cards ordered by player.
        state.apply_action(card)
    board = iter((20, 25, 30, 35, 40))
    for _ in range(street):
        state.apply_action(1)
        state.apply_action(1)
        while state.is_chance_node():
            state.apply_action(next(board))
    return state


@pytest.mark.parametrize("street", range(4))
def test_every_street_betting_sequence_against_reference_rules(street):
    """Enumerate all local betting paths, including limp/option and check-raise."""
    start = decision_state(street)
    encoder = HULHEFeatureEncoder()
    bet_size = 100 if street < 2 else 200
    leaves = []

    def walk(state, spent, bet_count, actions=()):
        player = state.current_player()
        other = 1 - player
        owed = spent[other] - spent[player]
        expected_legal = ([0] if owed else []) + [1] + ([2] if bet_count < 4 else [])
        assert state.legal_actions() == expected_legal
        np.testing.assert_array_equal(state.observation_tensor(player)[-2:], spent)
        encoded = encoder.information_state(state)
        np.testing.assert_array_equal(encoded[-3:], state.legal_actions_mask())
        assert np.isfinite(encoded).all() and encoded.min() >= 0 and encoded.max() <= 1
        for action in expected_legal:
            child = state.child(action)
            new_spent = spent.copy()
            if action == 0:
                assert child.is_terminal()
                assert child.returns()[player] == -spent[player]
                assert child.returns()[other] == spent[player]
                leaves.append(actions + (action,))
                continue
            if action == 2:
                new_spent[player] = spent[other] + bet_size
                walk(child, new_spent, bet_count + 1, actions + (action,))
                continue
            new_spent[player] = spent[other]
            # A small-blind completion leaves the big blind its option.
            ends_round = (owed > 0 and not (street == 0 and not actions)) or (
                owed == 0 and bool(actions) and actions[-1] == 1)
            if ends_round:
                if street == 3:
                    assert child.is_terminal()
                else:
                    assert child.is_chance_node()
                np.testing.assert_array_equal(child.observation_tensor(player)[-2:], new_spent)
                if child.is_terminal():
                    assert child.returns() == [new_spent[1], -new_spent[1]]  # AA wins.
                leaves.append(actions + (action,))
            else:
                assert child.current_player() == other
                walk(child, new_spent, bet_count, actions + (action,))

    walk(start, list(start.observation_tensor(0)[-2:]), 1 if street == 0 else 0)
    assert (1, 1) in leaves
    assert (1, 2, 1) in leaves
    capped_sequence = (2, 2, 2, 1) if street == 0 else (2, 2, 2, 2, 1)
    assert capped_sequence in leaves


def test_longest_capped_hand_and_payoff_units():
    state = decision_state()
    encoder = HULHEFeatureEncoder()
    board = iter((20, 25, 30, 35, 40))
    for street, commitment in enumerate((400, 800, 1600, 2400)):
        assert state.current_player() == (0 if street == 0 else 1)
        state.apply_action(1)  # Limp preflop or check postflop maximizes history.
        for _ in range(3 if street == 0 else 4):
            assert 2 in state.legal_actions()
            state.apply_action(2)
            encoder.information_state(state)
            encoder.full_state(state)
        assert state.legal_actions() == [0, 1]
        state.apply_action(1)
        assert state.observation_tensor(0)[-2:] == [commitment, commitment]
        while state.is_chance_node():
            state.apply_action(next(board))
    assert state.is_terminal()
    assert state.returns() == [2400.0, -2400.0]
    assert len(state.history()) == 32
    assert not encoder.full_state(state).any()
    game = load_hulhe_game()
    assert (game.min_utility(), game.max_utility()) == (-2400, 2400)
    assert (encoder.raw_size, encoder.policy_size, encoder.full_state_size) == (208, 417, 605)
    assert serialisable_game_definition()["small_bet_chips"] == 100


def test_folded_blind_and_split_pot():
    state = decision_state()
    state.apply_action(0)
    assert state.returns() == [-50.0, 50.0]
    # Everyone plays the royal flush on the board.
    board = iter((32, 36, 40, 44, 48))
    # Use private cards which do not collide with the board.
    state = load_hulhe_game().new_initial_state()
    for card in (0, 1, 2, 3):
        state.apply_action(card)
    while not state.is_terminal():
        state.apply_action(next(board) if state.is_chance_node() else 1)
    assert state.returns() == [0, 0]


def test_foreign_rules_and_legacy_encoders_are_rejected():
    encoder = HULHEFeatureEncoder()
    for changes in ({"maxRaises": "4 4 4 4"}, {"raiseSize": "100 100 100 100"},
                    {"firstPlayer": "2 1 1 1"}, {"betting": "nolimit"}):
        game = pyspiel.load_game("universal_poker", dict(HULHE_GAME_PARAMETERS, **changes))
        with pytest.raises(ValueError, match="contract"):
            encoder.information_state(decision_state(game=game))
    with pytest.raises(ValueError, match="Unsupported"):
        make_feature_encoder("hulhe_suit_canonical_fcpa_v1")


def test_checkpoint_and_continuation_reject_old_game_contract(tmp_path):
    payload = {"type": CHECKPOINT_TYPE, "version": 1,
               "game": serialisable_game_definition()}
    path = tmp_path / "policy.pkl"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint_payload(path)
    payload["version"] = CHECKPOINT_VERSION
    payload["game"]["parameters"]["betting"] = "nolimit"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="game definition"):
        load_checkpoint_payload(path)
    # Must reject before attempting to restore networks or replay.
    with pytest.raises(ValueError, match="game mismatch"):
        restore_training_state(None, {"game": deepcopy(payload["game"])})
    foreign = pyspiel.load_game("universal_poker", dict(HULHE_GAME_PARAMETERS, maxRaises="4 4 4 4"))
    with pytest.raises(ValueError, match="contract"):
        LoadedHULHEPolicy(foreign, path)
