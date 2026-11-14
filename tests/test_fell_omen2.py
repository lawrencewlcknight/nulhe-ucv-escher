"""Opt-in real Java/table integration. No downloads or training during tests."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hulhe_escher.fell_omen2 import FellOmen2Policy, observation_request, action_node
from hulhe_escher.fo2_setup import DEFAULT_ASSETS
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.evaluate import UniformPolicy, AlwaysCallPolicy, run_matches
from test_evaluate import decision

pytestmark = [pytest.mark.fo2, pytest.mark.skipif(os.getenv('RUN_FO2_TESTS') != '1',
                                               reason='Set RUN_FO2_TESTS=1 after fo2_setup')]


@pytest.fixture(scope='module')
def opponent():
    with FellOmen2Policy(load_hulhe_game(), os.getenv('FO2_ASSETS', DEFAULT_ASSETS)) as policy:
        yield policy


def test_all_local_branches_match_nodes_and_flag_upstream_free_folds(opponent):
    incompatible = []
    for street in range(4):
        def walk(state):
            if state.is_terminal() or state.is_chance_node():
                return
            try:
                probabilities = opponent.action_probabilities(state)
            except ValueError as exc:
                # Known upstream incompatibility is not hidden or normalized.
                raw = opponent.last_diagnostics['probabilities']
                assert 'illegal actions' in str(exc)
                assert 0 not in state.legal_actions() and raw[0] > 0
                incompatible.append(opponent.last_diagnostics['table'])
            else:
                assert set(probabilities) == set(state.legal_actions())
                assert sum(probabilities.values()) == pytest.approx(1)
            q = observation_request(state)
            assert opponent.last_diagnostics['node'] == action_node(street, q['actor'], q['raises'])
            for a in state.legal_actions(): walk(state.child(a))
        walk(decision(street))
    assert 'p1_strategy828080_5.txt' in incompatible


def test_original_reader_and_uncached_classification_match_bridge(opponent):
    fixture = json.loads((Path(__file__).parent/'fixtures/fo2_reference.json').read_text())
    for case in fixture['cases']:
        cached = opponent.request(case['request'])
        reference = opponent.request(case['request'], reference=True)
        assert cached == reference
        assert reference == case['expected']


def test_no_hidden_cards_and_no_cross_hand_state(opponent):
    a = decision(1)
    b = decision(1, holes=(1, 6, 10, 15))
    first = opponent.action_probabilities(a)
    assert first == opponent.action_probabilities(b)
    opponent.action_probabilities(decision(3, holes=(48, 49, 2, 3)))
    assert opponent.action_probabilities(a) == first


def test_reproducible_real_matches_in_both_seats(opponent, tmp_path):
    for name in ('a', 'b'):
        run_matches(UniformPolicy(), opponent, pairs=3, seed=14, output=tmp_path/name)
    assert (tmp_path/'a/hands.jsonl').read_bytes() == (tmp_path/'b/hands.jsonl').read_bytes()
    run_matches(AlwaysCallPolicy(), opponent, pairs=2, seed=3, output=tmp_path/'call')


def test_bad_request_closes_stream_instead_of_using_fallback():
    with FellOmen2Policy(load_hulhe_game(), os.getenv('FO2_ASSETS', DEFAULT_ASSETS)) as policy:
        request = observation_request(decision())
        request['hole'] = 'Ac Ac'
        with pytest.raises(RuntimeError, match='duplicate'):
            policy.request(request)
        with pytest.raises(RuntimeError, match='closed'):
            policy.request(observation_request(decision()))


def test_explicit_adaptation_preserves_raise_mass_and_logs_raw_strategy():
    with FellOmen2Policy(load_hulhe_game(), os.getenv('FO2_ASSETS', DEFAULT_ASSETS), free_fold='check') as policy:
        probabilities = policy.action_probabilities(decision(3))
        assert probabilities[1] == pytest.approx(.308218 + .426898)
        assert probabilities[2] == pytest.approx(.264884)
        assert policy.last_diagnostics['probabilities'][0] == .308218
        assert policy.last_diagnostics['legalization'] == 'explicit_free_fold_to_check'


def test_timeout_closes_process_and_history_disagreement_is_rejected():
    with FellOmen2Policy(load_hulhe_game(), os.getenv('FO2_ASSETS', DEFAULT_ASSETS), timeout=1e-9) as policy:
        with pytest.raises(TimeoutError): policy.request(observation_request(decision()))
        assert policy._process is None
    with FellOmen2Policy(load_hulhe_game(), os.getenv('FO2_ASSETS', DEFAULT_ASSETS)) as policy:
        request = observation_request(decision(1))
        request['preflop'] = 0
        with pytest.raises(RuntimeError, match='history differs'):
            policy.request(request)
