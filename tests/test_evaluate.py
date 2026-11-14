import json
from copy import deepcopy

import numpy as np
import pytest

from hulhe_escher.evaluate import AlwaysCallPolicy, UniformPolicy, play_hand, run_matches
from hulhe_escher.fell_omen2 import observation_request, card_string, validate_probabilities
from hulhe_escher.game import load_hulhe_game
from hulhe_escher import fo2_setup


def decision(street=0, holes=(0, 5, 10, 15), board=(20, 25, 30, 35, 40)):
    state = load_hulhe_game().new_initial_state()
    for card in holes:
        state.apply_action(card)
    cards = iter(board)
    for _ in range(street):
        state.apply_action(1)
        state.apply_action(1)
        while state.is_chance_node():
            state.apply_action(next(cards))
    return state


def test_public_observation_card_mapping_and_hidden_card_isolation():
    assert [card_string(c) for c in (0, 1, 2, 3, 48, 51)] == ['2c', '2d', '2h', '2s', 'Ac', 'As']
    for street in range(4):
        a = decision(street)
        player = a.current_player()
        holes = (1, 6, 10, 15) if player == 1 else (0, 5, 11, 14)
        b = decision(street, holes=holes)
        assert observation_request(a) == observation_request(b)
        q = observation_request(a)
        assert len(q['board'].split()) == (0, 3, 4, 5)[street]
        assert q['history'] == '/'.join(['cc'] * street + [''])
        assert q['preflop'] == (8 if street else -1)
        assert set(q) == {'hole', 'board', 'street', 'actor', 'raises', 'preflop', 'flop', 'turn', 'history'}
    with pytest.raises(ValueError, match='decision'):
        observation_request(load_hulhe_game().new_initial_state())


def test_history_records_both_raise_initiators_and_capped_rounds():
    # SB raise, BB call -> node 4 -> terminal index 3.
    s = decision()
    s.apply_action(2)
    s.apply_action(1)
    for c in (20, 25, 30): s.apply_action(c)
    assert observation_request(s)['preflop'] == 3
    # SB limp, BB raise, SB call -> node 9 -> terminal index 7.
    s = decision()
    for a in (1, 2, 1): s.apply_action(a)
    for c in (20, 25, 30): s.apply_action(c)
    assert observation_request(s)['preflop'] == 7
    # Three preflop raises reach four bets, ending at node 2 / terminal 1.
    s = decision()
    for a in (2, 2, 2, 1): s.apply_action(a)
    for c in (20, 25, 30): s.apply_action(c)
    assert observation_request(s)['preflop'] == 1


def test_probability_validation_fails_instead_of_substituting_an_action():
    assert sum(validate_probabilities([0, .333333, .666668], [1, 2]).values()) == pytest.approx(1)
    for probabilities, legal in [([.1, .4, .5], [1, 2]), ([0, .2, .3], [1, 2]),
                                  ([0, np.nan, 1], [1, 2]), ([-.1, .6, .5], [0, 1, 2]),
                                  ([0, 1, 0, 0], [1, 2])]:
        with pytest.raises(ValueError): validate_probabilities(probabilities, legal)


def test_duplicate_matches_repeat_without_touching_global_rng(tmp_path):
    np.random.seed(99)
    before = deepcopy(np.random.get_state())
    for name in ('a', 'b'):
        summary = run_matches(UniformPolicy(), AlwaysCallPolicy(), pairs=4, seed=13, output=tmp_path/name)
        assert summary['hands'] == 8
    after = np.random.get_state()
    np.testing.assert_equal(before, after)
    assert (tmp_path/'a/hands.jsonl').read_bytes() == (tmp_path/'b/hands.jsonl').read_bytes()
    records = [json.loads(line) for line in (tmp_path/'a/hands.jsonl').read_text().splitlines()]
    for a, b in zip(records[::2], records[1::2]):
        assert a['deal'] == b['deal']
        assert (a['candidate_seat'], b['candidate_seat']) == (0, 1)
        assert sum(a['returns_chips']) == sum(b['returns_chips']) == 0
    with pytest.raises(ValueError, match='empty'):
        run_matches(UniformPolicy(), AlwaysCallPolicy(), pairs=1, seed=13, output=tmp_path/'a')


def test_call_mirror_duplicate_score_cancels_and_uses_pairs_for_uncertainty(tmp_path):
    summary = run_matches(AlwaysCallPolicy(), AlwaysCallPolicy(), pairs=3, seed=1, output=tmp_path)
    assert summary['candidate_mean_small_bets_per_hand'] == 0
    assert summary['paired_standard_error'] == 0


def test_failures_are_logged_without_silent_fallback(tmp_path):
    class Broken:
        def action_probabilities(self, state):
            raise RuntimeError('missing strategy')
    with pytest.raises(RuntimeError, match='missing strategy'):
        run_matches(Broken(), AlwaysCallPolicy(), pairs=1, seed=0, output=tmp_path)
    assert json.loads((tmp_path/'failure.json').read_text())['completed_hands'] == 0
    assert not (tmp_path/'summary.json').exists()
    with pytest.raises(ValueError, match='distinct'):
        play_hand(load_hulhe_game(), [AlwaysCallPolicy()]*2, [0]*9, seed=0, pair=0, leg=0)


def test_missing_and_tampered_assets_fail_before_launch(monkeypatch, tmp_path):
    import hashlib
    monkeypatch.setattr(fo2_setup, 'asset_manifest', lambda: {'files': [
        {'path': 'table.zip', 'sha256': hashlib.sha256(b'expected').hexdigest()}]})
    with pytest.raises(ValueError, match='Missing or corrupt'):
        fo2_setup.verify_assets(tmp_path)
    (tmp_path/'table.zip').write_bytes(b'expected')
    fo2_setup.verify_assets(tmp_path)
    (tmp_path/'table.zip').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Missing or corrupt'):
        fo2_setup.verify_assets(tmp_path)
